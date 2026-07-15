from datetime import datetime
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

from config import VAPConfig, ProfilerConfig

import docker
from docker.types import Mount, Ulimit

logger = logging.getLogger("VAP")


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
            logging.StreamHandler(),  # 同时打到终端
        ],
        force=True,  # Python 3.8+，重复调用会覆盖旧配置
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

    requests.post(start_profile_url)
    exit_code, output = container.exec_run(
        ["/bin/bash", "-c", bench_cmd_str],
        demux=True,
    )
    requests.post(stop_profile_url)
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


def visualize_profile(config: VAPConfig, log_dir: str):
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
    tensorboard_cmd = [
        venv_python,
        "-m",
        "tensorboard.main",
        "--logdir",
        os.path.join(log_dir, "vllm-profile"),
    ]
    try:
        tensorboard_process = subprocess.Popen(tensorboard_cmd)
    except FileNotFoundError:
        logger.warning("%s is not available; skip TensorBoard visualization", venv_python)
        return
    logger.info("TensorBoard started with pid %s", tensorboard_process.pid)
    try:
        tensorboard_process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping TensorBoard...")
        raise
    finally:
        terminate_process(tensorboard_process)


def clean(log_dir: str):
    shutil.rmtree(log_dir)


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




if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run VAP")
    clean_parser = subparsers.add_parser("clean", help="Clean VAP")
    run_parser.add_argument("--config", type=str, default="example-config.json")
    args = argparser.parse_args()

    cwd = os.getcwd()
    log_dir = os.path.join(cwd, "logs")

    if args.command == "run":
        run(args, log_dir)
    elif args.command == "clean":
        clean(log_dir)
    else:
        argparser.print_help()
        exit(1)


