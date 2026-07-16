from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from config import VAPConfig


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "public"
DEFAULT_CONFIG_PATH = APP_DIR / "example-config.json"
WORK_DIR = Path.cwd().resolve()
CONFIG_PATH = WORK_DIR / "config.json"
LOGS_DIR = WORK_DIR / "logs"
SHELL_UNSAFE_PATTERN = re.compile(r"[\n\r;&|`$<>]")
ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RUN_LOCK = threading.Lock()
RUN_STATE: dict[str, Any] = {
    "process": None,
    "pid": None,
    "running": False,
    "exit_code": None,
    "started_at": None,
    "ended_at": None,
    "run_dir": None,
    "config_path": None,
    "output": "",
}


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        config = VAPConfig.model_validate(payload)
        runtime_errors = validate_runtime_config(config)
        if runtime_errors:
            return {
                "valid": False,
                "message": "Config validation failed.",
                "errors": runtime_errors,
            }
        return {
            "valid": True,
            "message": "Config is valid and can be used for a VAP run.",
            "summary": build_config_summary(config),
        }
    except Exception as exc:
        return {
            "valid": False,
            "message": "Config validation failed.",
            "errors": format_validation_error(exc),
        }


def validate_runtime_config(config: VAPConfig) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    deploy_host = config.vllm_deploy_cfg.get("--host")
    bench_host = config.vllm_bench_cfg.get("--host")
    deploy_port = config.vllm_deploy_cfg.get("--port")
    bench_port = config.vllm_bench_cfg.get("--port")

    if deploy_host is None:
        errors.append({"path": "vllm_deploy_cfg.--host", "message": "Missing vLLM deploy host"})
    if bench_host is None:
        errors.append({"path": "vllm_bench_cfg.--host", "message": "Missing vLLM benchmark host"})
    if deploy_host is not None and bench_host is not None and deploy_host != bench_host:
        errors.append(
            {
                "path": "vllm_deploy_cfg.--host",
                "message": "Deploy host must match benchmark host",
            }
        )

    if deploy_port is None:
        errors.append({"path": "vllm_deploy_cfg.--port", "message": "Missing vLLM deploy port"})
    if bench_port is None:
        errors.append({"path": "vllm_bench_cfg.--port", "message": "Missing vLLM benchmark port"})
    if deploy_port is not None and bench_port is not None and deploy_port != bench_port:
        errors.append(
            {
                "path": "vllm_deploy_cfg.--port",
                "message": "Deploy port must match benchmark port",
            }
        )
    if deploy_port is not None and not is_valid_port(deploy_port):
        errors.append(
            {
                "path": "vllm_deploy_cfg.--port",
                "message": "Port must be an integer from 1 to 65535",
            }
        )

    if config.distributed_cfg is not None:
        distributed = config.distributed_cfg
        if not is_valid_port(distributed.ray_port):
            errors.append(
                {
                    "path": "distributed_cfg.ray_port",
                    "message": "Ray port must be an integer from 1 to 65535",
                }
            )

    errors.extend(validate_risky_config(config))
    return errors


def validate_risky_config(config: VAPConfig) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    model_name = config.model_cfg.model_name
    if os.path.isabs(model_name) or ".." in Path(model_name).parts:
        errors.append(
            {
                "path": "model_cfg.model_name",
                "message": "Model name cannot be an absolute path or contain '..'",
            }
        )
    if has_shell_unsafe_chars(model_name):
        errors.append(
            {
                "path": "model_cfg.model_name",
                "message": "Model name contains shell-unsafe characters",
            }
        )

    image = config.docker_image
    if has_shell_unsafe_chars(image) or any(ch.isspace() for ch in image):
        errors.append(
            {
                "path": "container_cfg.image_name",
                "message": "Docker image name or tag cannot contain whitespace or shell-unsafe characters",
            }
        )

    for cfg_name, cfg in (
        ("vllm_deploy_cfg", config.vllm_deploy_cfg),
        ("vllm_bench_cfg", config.vllm_bench_cfg),
    ):
        errors.extend(validate_cli_args(cfg_name, cfg))

    for key, value in (config.container_cfg.env_vars or {}).items():
        if not ENV_KEY_PATTERN.match(key):
            errors.append(
                {
                    "path": f"container_cfg.env_vars.{key}",
                    "message": "Environment variable keys can only contain letters, digits, and underscores, and cannot start with a digit",
                }
            )
        if has_shell_unsafe_chars(value):
            errors.append(
                {
                    "path": f"container_cfg.env_vars.{key}",
                    "message": "Environment variable value contains newlines or shell-unsafe characters",
                }
            )

    for index, mount in enumerate(config.container_cfg.mounts or []):
        if not os.path.isabs(mount.source):
            errors.append(
                {
                    "path": f"container_cfg.mounts.{index}.source",
                    "message": "Host mount source must be an absolute path",
                }
            )
        if not os.path.isabs(mount.target):
            errors.append(
                {
                    "path": f"container_cfg.mounts.{index}.target",
                    "message": "Container mount target must be an absolute path",
                }
            )
        if has_shell_unsafe_chars(mount.source) or has_shell_unsafe_chars(mount.target):
            errors.append(
                {
                    "path": f"container_cfg.mounts.{index}",
                    "message": "Mount path contains shell-unsafe characters",
                }
            )

    for index, device in enumerate(config.container_cfg.devices or []):
        if not os.path.isabs(device):
            errors.append(
                {
                    "path": f"container_cfg.devices.{index}",
                    "message": "Device path must be an absolute path",
                }
            )
        if has_shell_unsafe_chars(device):
            errors.append(
                {
                    "path": f"container_cfg.devices.{index}",
                    "message": "Device path contains shell-unsafe characters",
                }
            )

    return errors


def validate_cli_args(cfg_name: str, cfg: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for key, value in cfg.items():
        if not key.startswith("-"):
            errors.append(
                {
                    "path": f"{cfg_name}.{key}",
                    "message": "CLI argument key must start with '-'",
                }
            )
        if has_shell_unsafe_chars(key) or any(ch.isspace() for ch in key):
            errors.append(
                {
                    "path": f"{cfg_name}.{key}",
                    "message": "CLI argument key cannot contain whitespace or shell-unsafe characters",
                }
            )
        if value is not None and isinstance(value, str):
            if has_shell_unsafe_chars(value):
                errors.append(
                    {
                        "path": f"{cfg_name}.{key}",
                        "message": "CLI argument value contains shell-unsafe characters",
                    }
                )
            if any(ch.isspace() for ch in value):
                errors.append(
                    {
                        "path": f"{cfg_name}.{key}",
                        "message": "CLI argument value does not currently support whitespace",
                    }
                )
    return errors


def has_shell_unsafe_chars(value: Any) -> bool:
    return isinstance(value, str) and bool(SHELL_UNSAFE_PATTERN.search(value))


def is_valid_port(port: Any) -> bool:
    return isinstance(port, int) and 1 <= port <= 65535


def format_validation_error(exc: Exception) -> list[dict[str, str]]:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        return [
            {
                "path": ".".join(str(part) for part in item.get("loc", [])) or "root",
                "message": item.get("msg", str(exc)),
            }
            for item in errors()
        ]
    return [{"path": "root", "message": str(exc)}]


def build_config_summary(config: VAPConfig) -> dict[str, Any]:
    distributed = config.distributed_cfg
    return {
        "model": config.model_cfg.model_name,
        "model_path": config.model_path,
        "docker_image": config.docker_image,
        "vllm_host": config.vllm_host,
        "vllm_port": config.vllm_port,
        "distributed": bool(distributed),
        "node_count": distributed.num_nodes if distributed else 1,
    }


def save_temp_config(payload: dict[str, Any]) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_path = WORK_DIR / f"vap-config-{timestamp}-{uuid.uuid4().hex[:8]}.json"
    with temp_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=4, ensure_ascii=False)
        config_file.write("\n")
    return temp_path


def is_local_port_available(port: int) -> bool:
    bind_targets = [
        (socket.AF_INET, "0.0.0.0"),
        (socket.AF_INET, "127.0.0.1"),
    ]
    if socket.has_ipv6:
        bind_targets.extend(
            [
                (socket.AF_INET6, "::"),
                (socket.AF_INET6, "::1"),
            ]
        )

    for family, host in bind_targets:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.bind((host, port))
            except OSError:
                return False
    return True


def check_config_ports(payload: dict[str, Any]) -> dict[str, Any]:
    validation = validate_config_payload(payload)
    if not validation["valid"]:
        return {"valid": False, "ports": [], "errors": validation["errors"]}

    config = VAPConfig.model_validate(payload)
    ports = [
        {
            "name": "vLLM service port",
            "port": config.vllm_port,
            "available": is_local_port_available(config.vllm_port),
        }
    ]
    if config.distributed_cfg is not None:
        ray_port = config.distributed_cfg.ray_port
        ports.append(
            {
                "name": "Ray distributed port",
                "port": ray_port,
                "available": is_local_port_available(ray_port),
            }
        )

    for item in ports:
        item["message"] = (
            f"Local port {item['port']} is available"
            if item["available"]
            else f"Local port {item['port']} is already in use or cannot be bound"
        )
    return {"valid": True, "ports": ports}


def check_config_machines(payload: dict[str, Any]) -> dict[str, Any]:
    validation = validate_config_payload(payload)
    if not validation["valid"]:
        return {"valid": False, "machines": [], "errors": validation["errors"]}

    config = VAPConfig.model_validate(payload)
    distributed = config.distributed_cfg
    if distributed is None:
        return {
            "valid": True,
            "machines": [],
            "message": "Distributed machines are not enabled in the current config.",
        }

    machines: list[dict[str, Any]] = []
    nodes = [distributed.head_node, *distributed.worker_nodes]
    for node in dict.fromkeys(nodes):
        checks = [{"label": "SSH", "port": 22}]
        if node == distributed.head_node:
            checks.append({"label": "Ray", "port": distributed.ray_port})
        machines.append(check_machine(node, checks))
    return {"valid": True, "machines": machines}


def check_config_resources(payload: dict[str, Any]) -> dict[str, Any]:
    model_cfg = payload.get("model_cfg") or {}
    container_cfg = payload.get("container_cfg") or {}
    model_root = str(model_cfg.get("model_path") or "")
    model_name = str(model_cfg.get("model_name") or "")
    image_name = str(container_cfg.get("image_name") or "")
    image_tag = str(container_cfg.get("image_tag") or "")
    model_path = str(Path(model_root) / model_name) if model_root and model_name else ""
    docker_image = f"{image_name}:{image_tag}" if image_name and image_tag else ""

    checks = [
        check_path(
            "Model root",
            model_root,
            expect_dir=True,
            required=True,
        ),
        check_path(
            "Model weight path",
            model_path,
            expect_dir=True,
            required=True,
        ),
        check_docker_image(docker_image),
    ]

    devices = container_cfg.get("devices") or []
    if not isinstance(devices, list):
        checks.append(
            {
                "name": "Device files",
                "ok": False,
                "message": "devices must be an array or null",
                "path": "",
            }
        )
        devices = []
    for index, device in enumerate(devices):
        checks.append(
            check_path(
                f"Device file devices[{index}]",
                str(device),
                expect_dir=False,
                required=True,
            )
        )

    mounts = container_cfg.get("mounts") or []
    if not isinstance(mounts, list):
        checks.append(
            {
                "name": "Mount sources",
                "ok": False,
                "message": "mounts must be an array or null",
                "path": "",
            }
        )
        mounts = []
    for index, mount in enumerate(mounts):
        if not isinstance(mount, dict):
            checks.append(
                {
                    "name": f"Mount source mounts[{index}].source",
                    "ok": False,
                    "message": "mount must be an object",
                    "path": "",
                }
            )
            continue
        checks.append(
            check_path(
                f"Mount source mounts[{index}].source",
                str(mount.get("source") or ""),
                expect_dir=None,
                required=True,
            )
        )

    return {"valid": True, "checks": checks}


def check_path(
    name: str,
    raw_path: str,
    *,
    expect_dir: bool | None,
    required: bool,
) -> dict[str, Any]:
    if not raw_path:
        return {
            "name": name,
            "ok": not required,
            "message": "Path is empty" if required else "Not configured, skipped",
            "path": raw_path,
        }

    path = Path(raw_path)
    if not path.is_absolute():
        return {
            "name": name,
            "ok": False,
            "message": "Path must be an absolute path",
            "path": raw_path,
        }
    if not path.exists():
        return {
            "name": name,
            "ok": False,
            "message": "Path does not exist",
            "path": raw_path,
        }
    if expect_dir is True and not path.is_dir():
        return {
            "name": name,
            "ok": False,
            "message": "Path exists but is not a directory",
            "path": raw_path,
        }
    if expect_dir is False and not path.exists():
        return {
            "name": name,
            "ok": False,
            "message": "File or device does not exist",
            "path": raw_path,
        }
    return {
        "name": name,
        "ok": True,
        "message": "Exists and is accessible",
        "path": raw_path,
    }


def check_docker_image(image: str) -> dict[str, Any]:
    if not image:
        return {
            "name": "Local Docker image",
            "ok": False,
            "message": "Docker image config is empty",
            "image": image,
        }
    try:
        import docker
        from docker.errors import DockerException, ImageNotFound
    except Exception as exc:
        return {
            "name": "Local Docker image",
            "ok": False,
            "message": f"Cannot import Docker SDK: {exc}",
            "image": image,
        }

    try:
        client = docker.from_env()
        client.images.get(image)
        return {
            "name": "Local Docker image",
            "ok": True,
            "message": "Local Docker image exists",
            "image": image,
        }
    except ImageNotFound:
        return {
            "name": "Local Docker image",
            "ok": False,
            "message": "Local Docker image does not exist. Pull or build it first.",
            "image": image,
        }
    except DockerException as exc:
        return {
            "name": "Local Docker image",
            "ok": False,
            "message": f"Docker daemon is unavailable or permission was denied: {exc}",
            "image": image,
        }


def check_machine(node: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        resolved_ip = socket.gethostbyname(node)
    except OSError as exc:
        return {
            "node": node,
            "reachable": False,
            "ip": None,
            "checks": [],
            "message": f"DNS resolution failed: {exc}",
        }

    port_results = []
    for check in checks:
        port = int(check["port"])
        label = str(check["label"])
        try:
            with socket.create_connection((node, port), timeout=1.5):
                port_results.append(
                    {
                        "label": label,
                        "port": port,
                        "reachable": True,
                        "message": f"{label} port {port} is reachable",
                    }
                )
        except OSError as exc:
            port_results.append(
                {
                    "label": label,
                    "port": port,
                    "reachable": False,
                    "message": f"{label} port {port} is unreachable: {exc}",
                }
            )

    reachable = any(result["reachable"] for result in port_results)
    return {
        "node": node,
        "reachable": reachable,
        "ip": resolved_ip,
        "checks": port_results,
        "message": "Machine is reachable" if reachable else "Machine DNS resolved, but none of the checked ports are reachable",
    }


def resolve_config_path(raw_path: str | None) -> Path:
    if not raw_path:
        return DEFAULT_CONFIG_PATH
    candidate = (WORK_DIR / raw_path).resolve()
    if not candidate.is_relative_to(WORK_DIR):
        raise ValueError("Config path must be under the current working directory")
    return candidate


def get_run_state_snapshot() -> dict[str, Any]:
    with RUN_LOCK:
        process = RUN_STATE["process"]
        run_dir = RUN_STATE["run_dir"]
        return {
            "pid": RUN_STATE["pid"],
            "running": RUN_STATE["running"],
            "exit_code": RUN_STATE["exit_code"],
            "started_at": RUN_STATE["started_at"],
            "ended_at": RUN_STATE["ended_at"],
            "run_dir": str(run_dir) if run_dir else None,
            "config_path": str(RUN_STATE["config_path"]) if RUN_STATE["config_path"] else None,
            "output": RUN_STATE["output"],
            "has_process": process is not None,
        }


def read_current_log_file(file_name: str) -> dict[str, Any]:
    allowed_names = {"vap_log.txt", "vllm_deploy.log", "vllm_bench.log"}
    if file_name not in allowed_names:
        raise ValueError("Unsupported log file")

    snapshot = get_run_state_snapshot()
    run_dir = Path(snapshot["run_dir"]) if snapshot["run_dir"] else None
    if run_dir is None:
        return {
            "exists": False,
            "name": file_name,
            "path": None,
            "run_dir": None,
            "content": "",
            "message": "There is no active run or this run directory has not been created yet",
        }

    log_path = (run_dir / file_name).resolve()
    if log_path.is_file() and log_path.is_relative_to(LOGS_DIR.resolve()):
        return {
            "exists": True,
            "name": file_name,
            "path": str(log_path),
            "run_dir": str(run_dir),
            "content": log_path.read_text(encoding="utf-8", errors="replace"),
        }

    return {
        "exists": False,
        "name": file_name,
        "path": str(log_path),
        "run_dir": str(run_dir),
        "content": "",
        "message": f"This run has not generated {file_name} yet",
    }


def discover_run_dir(existing_dirs: set[str], started_at: float) -> Path | None:
    if not LOGS_DIR.is_dir():
        return None
    candidates = []
    for path in LOGS_DIR.iterdir():
        if not path.is_dir() or path.name in existing_dirs:
            continue
        try:
            if path.stat().st_mtime >= started_at - 1:
                candidates.append(path)
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def monitor_run_process(process: subprocess.Popen[str], existing_dirs: set[str], started_at: float) -> None:
    while True:
        with RUN_LOCK:
            if RUN_STATE["process"] is process and RUN_STATE["run_dir"] is None:
                RUN_STATE["run_dir"] = discover_run_dir(existing_dirs, started_at)
        line = process.stdout.readline() if process.stdout else ""
        if line:
            with RUN_LOCK:
                if RUN_STATE["process"] is process:
                    RUN_STATE["output"] += line
        elif process.poll() is not None:
            break
        else:
            time.sleep(0.2)

    remaining = process.stdout.read() if process.stdout else ""
    exit_code = process.wait()
    with RUN_LOCK:
        if RUN_STATE["process"] is not process:
            return
        if remaining:
            RUN_STATE["output"] += remaining
        if RUN_STATE["run_dir"] is None:
            RUN_STATE["run_dir"] = discover_run_dir(existing_dirs, started_at)
        RUN_STATE["running"] = False
        RUN_STATE["exit_code"] = exit_code
        RUN_STATE["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def current_run_is_tensorboard_phase() -> bool:
    snapshot = get_run_state_snapshot()
    tensorboard_marker = "TensorBoard started with pid"
    if tensorboard_marker in (snapshot.get("output") or ""):
        return True

    run_dir = Path(snapshot["run_dir"]) if snapshot.get("run_dir") else None
    if run_dir is None:
        return False
    log_path = (run_dir / "vap_log.txt").resolve()
    if not log_path.is_file() or not log_path.is_relative_to(LOGS_DIR.resolve()):
        return False
    return tensorboard_marker in log_path.read_text(encoding="utf-8", errors="replace")


def stop_process_group_sync(process: subprocess.Popen[str], timeout_sec: float = 5.0) -> bool:
    terminate_run_process(process)
    try:
        process.wait(timeout=timeout_sec)
        return True
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        process.kill()
    try:
        process.wait(timeout=timeout_sec)
        return True
    except subprocess.TimeoutExpired:
        return False


def start_vap_run(config_path: Path | None = None) -> dict[str, Any]:
    with RUN_LOCK:
        process = RUN_STATE["process"]
        is_running = RUN_STATE["running"] and process is not None and process.poll() is None
    if is_running:
        if not current_run_is_tensorboard_phase():
            raise RuntimeError("VAP is already running")
        with RUN_LOCK:
            RUN_STATE["output"] += "\n--- Previous TensorBoard is still running; stopping it before new run ---\n"
        if not stop_process_group_sync(process):
            raise RuntimeError("The previous TensorBoard process did not stop in time. Try again later.")

    run_config_path = (config_path or CONFIG_PATH).resolve()
    LOGS_DIR.mkdir(exist_ok=True)
    existing_dirs = {path.name for path in LOGS_DIR.iterdir() if path.is_dir()}
    started_at = time.time()
    process = subprocess.Popen(
        [sys.executable, "main.py", "run", "--config", str(run_config_path)],
        cwd=str(WORK_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    with RUN_LOCK:
        RUN_STATE.update(
            {
                "process": process,
                "pid": process.pid,
                "running": True,
                "exit_code": None,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ended_at": None,
                "run_dir": None,
                "config_path": run_config_path,
                "output": f"--- VAP started (pid {process.pid}) ---\n",
            }
        )
    thread = threading.Thread(
        target=monitor_run_process,
        args=(process, existing_dirs, started_at),
        daemon=True,
    )
    thread.start()
    return get_run_state_snapshot()


def terminate_run_process(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()


def force_kill_process_group_later(process: subprocess.Popen[str], timeout_sec: float = 5.0) -> None:
    def worker() -> None:
        try:
            process.wait(timeout=timeout_sec)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()

    threading.Thread(target=worker, daemon=True).start()


def stop_vap_run() -> dict[str, Any]:
    with RUN_LOCK:
        process = RUN_STATE["process"]
    if process is None or process.poll() is not None:
        return {"message": "There is no active VAP run", **get_run_state_snapshot()}
    terminate_run_process(process)
    force_kill_process_group_later(process)
    with RUN_LOCK:
        RUN_STATE["output"] += "\n--- Stop requested from UI ---\n"
    return {"message": "Stop signal sent. main.py and its child processes will be stopped.", **get_run_state_snapshot()}


class VAPConfigHandler(BaseHTTPRequestHandler):
    server_version = "VAPConfigServer/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_static("index.html")
            return
        if parsed.path == "/api/config":
            self.handle_get_config(parsed.query)
            return
        if parsed.path == "/api/log-file":
            self.handle_get_log_file(parsed.query)
            return
        if parsed.path == "/api/run/status":
            self.handle_run_status()
            return
        if parsed.path.startswith("/public/"):
            self.serve_static(parsed.path.removeprefix("/public/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/config":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self.handle_put_config()
        except json.JSONDecodeError as exc:
            self.send_json(
                {"message": f"JSON parse failed: {exc}"}, HTTPStatus.BAD_REQUEST
            )
        except Exception as exc:
            self.send_json(
                {"message": f"Failed to save config: {exc}"}, HTTPStatus.BAD_REQUEST
            )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        routes = {
            "/api/validate": self.handle_validate,
            "/api/temp-config": self.handle_save_temp_config,
            "/api/check-ports": self.handle_check_ports,
            "/api/check-machines": self.handle_check_machines,
            "/api/check-resources": self.handle_check_resources,
            "/api/run": self.handle_run_start,
            "/api/run/stop": self.handle_run_stop,
        }
        handler = routes.get(parsed.path)
        if handler is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            handler()
        except json.JSONDecodeError as exc:
            self.send_json(
                {"message": f"JSON parse failed: {exc}"}, HTTPStatus.BAD_REQUEST
            )
        except ValueError as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json(
                {"message": f"Server handling failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_get_config(self, query: str) -> None:
        try:
            params = parse_qs(query)
            raw_path = params.get("path", [None])[0]
            config_path = resolve_config_path(raw_path)
            with config_path.open("r", encoding="utf-8") as config_file:
                payload = json.load(config_file)
            self.send_json({"path": str(config_path), "config": payload})
        except Exception as exc:
            self.send_json(
                {"message": f"Failed to read config: {exc}"}, HTTPStatus.BAD_REQUEST
            )

    def handle_put_config(self) -> None:
        payload = self.read_json_body()
        validation = validate_config_payload(payload)
        if not validation["valid"]:
            self.send_json(
                {
                    "message": "Config validation failed before saving",
                    "errors": validation["errors"],
                },
                HTTPStatus.BAD_REQUEST,
            )
            return
        with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
            json.dump(payload, config_file, indent=4, ensure_ascii=False)
            config_file.write("\n")
        self.send_json(
            {
                "message": "Config saved",
                "path": str(CONFIG_PATH),
                "validation": validation,
            }
        )

    def handle_get_log_file(self, query: str) -> None:
        try:
            params = parse_qs(query)
            file_name = params.get("name", [""])[0]
            self.send_json(read_current_log_file(file_name))
        except Exception as exc:
            self.send_json(
                {"message": f"Failed to read log: {exc}"}, HTTPStatus.BAD_REQUEST
            )

    def handle_run_status(self) -> None:
        self.send_json(
            {
                **get_run_state_snapshot(),
                "logs": {
                    name: read_current_log_file(name)
                    for name in ("vap_log.txt", "vllm_deploy.log", "vllm_bench.log")
                },
            }
        )

    def handle_validate(self) -> None:
        payload = self.read_json_body()
        self.send_json(validate_config_payload(payload))

    def handle_save_temp_config(self) -> None:
        payload = self.read_json_body()
        temp_path = save_temp_config(payload)
        validation = validate_config_payload(payload)
        self.send_json(
            {
                "path": str(temp_path),
                "file_name": temp_path.name,
                "validation": validation,
                "message": "Temporary config file generated.",
            }
        )

    def handle_check_ports(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_ports(payload))

    def handle_check_machines(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_machines(payload))

    def handle_check_resources(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_resources(payload))

    def handle_run_start(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        config_path = None
        if length > 0:
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Run config must be a JSON object")
            validation = validate_config_payload(payload)
            if not validation["valid"]:
                self.send_json(
                    {
                        "message": "Run config validation failed",
                        "errors": validation["errors"],
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return
            config_path = save_temp_config(payload)
        self.send_json(start_vap_run(config_path))

    def handle_run_stop(self) -> None:
        self.send_json(stop_vap_run())

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def serve_static(self, relative_path: str) -> None:
        static_path = (STATIC_DIR / relative_path).resolve()
        if not static_path.is_relative_to(STATIC_DIR) or not static_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content = static_path.read_bytes()
        content_type = "text/html; charset=utf-8"
        if static_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif static_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[VAP Config UI] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VAP config management UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VAPConfigHandler)
    print(f"VAP config UI started: http://{args.host}:{args.port}")
    print(f"Temporary config files will be saved to: {WORK_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
