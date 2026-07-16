from datetime import datetime
import glob
import json
import os
import logging
import socket
import time
import requests
import argparse
import signal
import shutil
import subprocess

from config import VAPConfig
from runtime_paths import (
    APP_DIR,
    VAP_BIN_DIR,
    VAP_LOGS_DIR,
    VAP_PERFETTO_HOME,
    ensure_vap_home,
)

import docker
from docker.types import Mount, Ulimit

logger = logging.getLogger("VAP")
PERFETTO_PORT = 9001


def build_container_mounts(config: VAPConfig, log_path: str) -> list[Mount]:
    mounts: list[Mount] = []
    if config.container_cfg.mounts:
        for m in config.container_cfg.mounts:
            mounts.append(
                Mount(target=m.target, source=m.source, type=m.type or "bind")
            )
    mounts.extend(
        [
            Mount(
                target="/tmp/vap/models",
                source=config.model_cfg.model_path,
                type="bind",
            ),
            Mount(target="/app/VAP/log", source=log_path, type="bind"),
            Mount(
                target="/app/VAP/log/vllm-profile",
                source=os.path.join(log_path, "vllm-profile"),
                type="bind",
            ),
        ]
    )
    return mounts


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config_json = json.load(f)
    config = VAPConfig.model_validate_json(json.dumps(config_json))
    logger.info(f"Config loaded: {config}")
    config.distributed_cfg = None
    logger.warning("Distributed config is set to None manually for now.")

    return config


def setup_logging(log_path: str, debug: bool = False) -> logging.Logger:
    log_file = os.path.join(log_path, "vap_log.txt")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),  # Also log to terminal
        ],
        force=True,  # Python 3.8+: repeated calls replace the old config
    )
    return logging.getLogger("VAP")


def init_docker_client() -> docker.DockerClient:
    client = docker.from_env()
    return client


def is_machine_connected(node: str) -> bool:
    logger.warning("Checking machine connection is not implemented yet.")
    pass


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def check_port_availability(config: VAPConfig):
    if config.distributed_cfg is not None:
        ray_port = config.distributed_cfg.ray_port
        if is_port_available(ray_port):
            logger.info(f"Port {ray_port} is available")
        else:
            logger.error(f"Port {ray_port} is not available")
            raise Exception(f"Port {ray_port} is not available")

    vllm_port = config.vllm_port
    if is_port_available(vllm_port):
        logger.info(f"Port {vllm_port} is available")
    else:
        logger.error(f"Port {vllm_port} is not available")
        raise Exception(f"Port {vllm_port} is not available")


def check_remote_assets(node: str, asset_path: str) -> bool:
    pass
    logging.warning("Checking remote assets is not implemented yet.")


def wait_for_vllm_ready(
    vllm_port: int, timeout_sec: float = 1800, poll_interval_sec: float = 5
):
    """Wait until vLLM /health returns 200.

    Port listening and server ready are different: vLLM may bind the port only
    after model load, torch.compile, and CUDA graph capture complete.
    """
    url = f"http://127.0.0.1:{vllm_port}/health"
    deadline = time.monotonic() + timeout_sec
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("vLLM ready at %s (attempt %d)", url, attempt)
                return
            logger.info(
                "Waiting for vLLM: %s returned HTTP %d (attempt %d)",
                url,
                response.status_code,
                attempt,
            )
        except requests.exceptions.ConnectionError:
            logger.info(
                "Waiting for vLLM: port %d not accepting connections yet (attempt %d)",
                vllm_port,
                attempt,
            )
        except requests.exceptions.RequestException as exc:
            logger.info(
                "Waiting for vLLM: %s (attempt %d)",
                exc,
                attempt,
            )
        time.sleep(poll_interval_sec)

    raise TimeoutError(f"vLLM did not become ready at {url} within {timeout_sec:.0f}s")


def deploy_vllm_server(
    config: VAPConfig,
    log_path: str,
    docker_client: docker.DockerClient,
    full_image_name: str,
    date_str: str,
) -> docker.models.containers.Container:
    mounts = build_container_mounts(config, log_path)
    vllm_deploy_args = config.vllm_deploy_args_str()

    container_model_path = os.path.join("/tmp/vap/models", config.model_cfg.model_name)
    vllm_serve_cmd = (
        f"vllm serve {container_model_path} {vllm_deploy_args} "
        f"> /app/VAP/log/vllm_deploy.log 2>&1"
    )
    logger.debug("VLLM deploy command: %s", vllm_serve_cmd)

    safe_model_name = config.model_cfg.model_name.replace("/", "_")
    container_name = f"vap_{safe_model_name}_{date_str}"

    rocm_devices = [
        "/dev/kfd",
        "/dev/mem",
    ]
    devices = list(
        dict.fromkeys(rocm_devices + (config.container_cfg.devices or ["/dev/dri/"]))
    )
    os.makedirs(os.path.join(log_path, "vllm-profile"), exist_ok=True)

    container = docker_client.containers.run(
        image=full_image_name,
        name=container_name,
        ipc_mode="host",
        network_mode="host",
        cap_add=["SYS_ADMIN", "SYS_PTRACE"],
        devices=devices,
        ulimits=[
            Ulimit(name="memlock", soft=-1, hard=-1),
            Ulimit(name="nofile", soft=65535, hard=65535),
        ],
        shm_size="128G",
        group_add=["video"],
        security_opt=["seccomp=unconfined"],
        mounts=mounts,
        environment=config.container_cfg.env_vars or {},
        entrypoint=[],
        command=["/bin/bash", "-c", vllm_serve_cmd],
        detach=True,
        remove=False,
    )
    logger.info("Started vLLM container %s (%s)", container_name, container.id)
    return container


def bench_and_profile(
    config: VAPConfig,
    container: docker.models.containers.Container,
) -> None:
    if container is None:
        raise ValueError("container is required to run vllm bench inside docker")

    bench_cmd_str = (
        f"vllm bench serve {config.vllm_bench_args_str()} "
        f"2>&1 | tee /app/VAP/log/vllm_bench.log"
    )
    logger.debug("Benchmark and profile command: %s", bench_cmd_str)

    start_profile_url = f"http://{config.vllm_host}:{config.vllm_port}/start_profile"
    stop_profile_url = f"http://{config.vllm_host}:{config.vllm_port}/stop_profile"

    requests.post(start_profile_url, timeout=10)
    try:
        exit_code, output = container.exec_run(
            ["/bin/bash", "-c", bench_cmd_str],
            demux=True,
        )
    finally:
        try:
            requests.post(stop_profile_url, timeout=10)
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed to stop vLLM profiler cleanly: %s", exc)
    stdout, stderr = output
    if exit_code != 0:
        msg = (stderr or stdout or b"").decode(errors="replace")
        logger.error("Benchmark failed (exit %s): %s", exit_code, msg)
        raise RuntimeError(f"vllm bench failed with exit code {exit_code}")
    logger.info("Benchmark finished successfully")


def terminate_process(process: subprocess.Popen | None, timeout_sec: float = 5.0):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def write_visualization_pids(
    log_dir: str, processes: dict[str, subprocess.Popen | None]
) -> None:
    payload = {
        name: process.pid
        for name, process in processes.items()
        if process is not None and process.poll() is None
    }
    path = os.path.join(log_dir, "visualization_pids.json")
    with open(path, "w", encoding="utf-8") as pid_file:
        json.dump(payload, pid_file, indent=2)
        pid_file.write("\n")


def collect_pytorch_trace_files(profile_dir: str) -> list[str]:
    patterns = ("*.pt.trace.json.gz", "*.trace.json.gz", "*.trace.json")
    traces: list[str] = []
    for pattern in patterns:
        traces.extend(glob.glob(os.path.join(profile_dir, pattern)))
    traces = [
        trace for trace in traces if "merged_trace" not in os.path.basename(trace)
    ]
    return sorted(dict.fromkeys(traces))


def safe_filename_part(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value
    ).strip("_")


def merged_trace_output_file(profile_dir: str, config: VAPConfig) -> str:
    run_stamp = os.path.basename(os.path.dirname(profile_dir.rstrip(os.sep)))
    model_name = safe_filename_part(config.model_cfg.model_name.replace("/", "_"))
    prefix = "-".join(part for part in (run_stamp, model_name) if part)
    return os.path.join(profile_dir, f"{prefix}-merged_trace.json")


def merge_pytorch_traces_for_perfetto(
    profile_dir: str, config: VAPConfig
) -> str | None:
    trace_files = collect_pytorch_trace_files(profile_dir)
    if not trace_files:
        return None
    if len(trace_files) == 1:
        logger.info(
            "Only one PyTorch trace found; Perfetto will load %s", trace_files[0]
        )
        return trace_files[0]

    output_file = merged_trace_output_file(profile_dir, config)
    try:
        from TraceLens import TraceFuse

        logger.info("Merging %d PyTorch traces with TraceLens", len(trace_files))
        merged_trace = TraceFuse(trace_files).merge_and_save(output_file)
        logger.info("Merged Perfetto trace has been saved to: %s", merged_trace)
        return merged_trace
    except Exception as exc:
        logger.warning("TraceLens trace merge failed: %s", exc)
        logger.warning("Perfetto will fall back to the first trace: %s", trace_files[0])
        return trace_files[0]


def find_perfetto_trace(profile_dir: str, config: VAPConfig) -> str | None:
    merged_or_single_trace = merge_pytorch_traces_for_perfetto(profile_dir, config)
    if merged_or_single_trace:
        return merged_or_single_trace

    pftrace_files = sorted(glob.glob(os.path.join(profile_dir, "*.pftrace")))
    if pftrace_files:
        return pftrace_files[0]
    return None


def find_trace_processor(app_dir: str) -> str | None:
    candidates = [
        str(VAP_BIN_DIR / "trace_processor"),
        os.path.join(app_dir, "bin", "trace_processor"),
        os.path.join(app_dir, "trace_processor"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def visualize_profile(config: VAPConfig, log_dir: str):
    app_dir = str(APP_DIR)
    profile_dir = os.path.join(log_dir, "vllm-profile")
    venv_python = os.path.join(app_dir, ".venv", "bin", "python")
    tensorboard_cmd = [
        venv_python,
        "-m",
        "tensorboard.main",
        "--logdir",
        profile_dir,
        "--port",
        str(config.profiler_cfg.tensorboard_port),
    ]

    tensorboard_process = None
    perfetto_process = None
    try:
        tensorboard_process = subprocess.Popen(tensorboard_cmd)
    except FileNotFoundError:
        logger.warning(
            "%s is not available; skip TensorBoard visualization", venv_python
        )
    else:
        logger.info(
            "TensorBoard started with pid %s on port %s",
            tensorboard_process.pid,
            config.profiler_cfg.tensorboard_port,
        )

    trace_path = find_perfetto_trace(profile_dir, config)
    trace_processor = find_trace_processor(app_dir)
    if trace_path is None:
        logger.warning("No Perfetto-compatible trace found under %s", profile_dir)
    elif trace_processor is None:
        logger.warning("trace_processor is not available; skip Perfetto visualization")
    else:
        perfetto_home = str(VAP_PERFETTO_HOME)
        os.makedirs(perfetto_home, exist_ok=True)
        perfetto_env = os.environ.copy()
        perfetto_env["HOME"] = perfetto_home
        perfetto_cmd = [
            trace_processor,
            "--httpd",
            "--http-port",
            str(PERFETTO_PORT),
            trace_path,
        ]
        try:
            perfetto_process = subprocess.Popen(perfetto_cmd, env=perfetto_env)
        except FileNotFoundError:
            logger.warning(
                "%s is not available; skip Perfetto visualization", trace_processor
            )
        else:
            logger.info(
                "Perfetto Trace Processor started with pid %s on port %s for %s",
                perfetto_process.pid,
                PERFETTO_PORT,
                trace_path,
            )

    processes = [
        process for process in (tensorboard_process, perfetto_process) if process
    ]
    write_visualization_pids(
        log_dir,
        {"tensorboard": tensorboard_process, "perfetto": perfetto_process},
    )
    if not processes:
        return

    try:
        while any(process.poll() is None for process in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping visualization services...")
        raise
    finally:
        for process in processes:
            terminate_process(process)
        write_visualization_pids(log_dir, {})


def clean(log_dir: str):
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
        print(f"Removed {log_dir}")
    else:
        print(f"{log_dir} does not exist; nothing to clean")


def register_signal_handler(container: docker.models.containers.Container):
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} received. Cleaning up...")
        if container is not None:
            container.stop()
            container.remove()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def run(args, log_dir: str):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, date_str)
    os.makedirs(log_path, exist_ok=True)
    shutil.copy2(args.config, os.path.join(log_path, "config.json"))
    container = None

    logger = setup_logging(
        log_path, debug=True if os.getenv("VAP_DEBUG") == "1" else False
    )
    docker_client = init_docker_client()
    config = load_config(args.config)

    logger.info("VAP started")

    # 1. resource validation
    # 1.1. machine connection
    if config.distributed_cfg is not None:
        for node in config.distributed_cfg.worker_nodes:
            if not is_machine_connected(node):
                logger.error(f"Machine {node} is not connected")
                raise Exception(f"Machine {node} is not connected")
            logger.info(f"Machine {node} is connected")

    # 1.2. port availability
    check_port_availability(config)
    # 1.3. model weight availability
    if not os.path.exists(config.model_path):
        logger.error(f"Model weight {config.model_path} is not available")
        raise Exception(f"Model weight {config.model_path} is not available")
    logger.info(f"Model weight {config.model_path} is available")

    if config.distributed_cfg is not None:
        for node in config.distributed_cfg.worker_nodes:
            if not check_remote_assets(node, config.model_path):
                logger.error(
                    f"Model weight {config.model_path} is not available on machine {node}"
                )
                raise Exception(
                    f"Model weight {config.model_path} is not available on machine {node}"
                )
            logger.info(
                f"Model weight {config.model_path} is available on machine {node}"
            )
    # 1.4. image availability
    try:
        docker_client.images.get(config.docker_image)
    except docker.errors.ImageNotFound:
        logger.error("Docker image %s is not available", config.docker_image)
        raise
    logger.info("Docker image %s is available", config.docker_image)

    try:
        # 2. vllm server deployment
        container = deploy_vllm_server(
            config, log_path, docker_client, config.docker_image, date_str
        )
        register_signal_handler(container)
        vllm_port = config.vllm_port
        wait_for_vllm_ready(vllm_port)

        # 3. benchmark and profile
        bench_and_profile(config, container)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        if container is not None:
            container.stop()
            container.remove()

    # 4. print profile result path
    logger.info(
        f"Profile archive has been saved to: {os.path.join(log_path, 'vllm-profile')}"
    )
    # 5. visualization
    visualize_profile(config, log_path)


def main(argv: list[str] | None = None) -> None:
    ensure_vap_home()
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run VAP")
    subparsers.add_parser("clean", help="Clean VAP")
    run_parser.add_argument("--config", type=str, default="example-config.json")
    args = argparser.parse_args(argv)

    log_dir = str(VAP_LOGS_DIR)

    if args.command == "run":
        run(args, log_dir)
    elif args.command == "clean":
        clean(log_dir)
    else:
        argparser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
