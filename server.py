from __future__ import annotations

import argparse
import atexit
from collections import Counter, defaultdict
import gzip
import io
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
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent_runtime import AgentTool, VAPAgentRuntime
from config import VAPConfig
from runtime_paths import (
    APP_DIR,
    VAP_BIN_DIR,
    VAP_CONFIG_PATH,
    VAP_HOME,
    VAP_LOGS_DIR,
    VAP_PERFETTO_HOME,
    VAP_TEMP_CONFIG_DIR,
    ensure_vap_home,
    resolve_under_vap_home,
)

STATIC_DIR = APP_DIR / "public"
DEFAULT_CONFIG_PATH = APP_DIR / "example-config.json"
CONFIG_PATH = VAP_CONFIG_PATH
LOGS_DIR = VAP_LOGS_DIR
TEMP_CONFIG_DIR = VAP_TEMP_CONFIG_DIR
SHELL_UNSAFE_PATTERN = re.compile(r"[\n\r;&|`$<>]")
ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_VISIBLE_DEVICE_COUNT = 8
PERFETTO_PORT = 9001
SERVER_SESSION_ID = uuid.uuid4().hex
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
    "stop_requested": False,
}
AGENT_RUNTIME = VAPAgentRuntime()
AGENT_TOOLS_REGISTERED = False
AGENT_TOOLS_LOCK = threading.Lock()
SHUTDOWN_CLEANUP_LOCK = threading.Lock()
SHUTDOWN_CLEANUP_DONE = False
TORCHPROFILER_SKILL_DIR = APP_DIR / "skills" / "TorchProfilerTraceSkill"


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
        errors.append(
            {"path": "vllm_deploy_cfg.--host", "message": "Missing vLLM deploy host"}
        )
    if bench_host is None:
        errors.append(
            {"path": "vllm_bench_cfg.--host", "message": "Missing vLLM benchmark host"}
        )
    if deploy_host is not None and bench_host is not None and deploy_host != bench_host:
        errors.append(
            {
                "path": "vllm_deploy_cfg.--host",
                "message": "Deploy host must match benchmark host",
            }
        )

    if deploy_port is None:
        errors.append(
            {"path": "vllm_deploy_cfg.--port", "message": "Missing vLLM deploy port"}
        )
    if bench_port is None:
        errors.append(
            {"path": "vllm_bench_cfg.--port", "message": "Missing vLLM benchmark port"}
        )
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

    if not is_valid_port(config.profiler_cfg.tensorboard_port):
        errors.append(
            {
                "path": "profiler_cfg.tensorboard_port",
                "message": "TensorBoard port must be an integer from 1 to 65535",
            }
        )

    local_vllm_port = (
        deploy_port
        if is_valid_port(deploy_port) and deploy_port == bench_port
        else None
    )
    errors.extend(validate_local_service_port_conflicts(config, local_vllm_port))
    errors.extend(validate_tensor_parallel_devices(config))
    errors.extend(validate_risky_config(config))
    return errors


def validate_local_service_port_conflicts(
    config: VAPConfig,
    local_vllm_port: int | None,
) -> list[dict[str, str]]:
    ports = [
        (
            "profiler_cfg.tensorboard_port",
            "TensorBoard port",
            config.profiler_cfg.tensorboard_port,
        ),
        ("perfetto.port", "Perfetto Trace Processor port", PERFETTO_PORT),
    ]
    if local_vllm_port is not None:
        ports.insert(
            0, ("vllm_deploy_cfg.--port", "vLLM service port", local_vllm_port)
        )
    if config.distributed_cfg is not None:
        ports.append(
            (
                "distributed_cfg.ray_port",
                "Ray distributed port",
                config.distributed_cfg.ray_port,
            )
        )

    errors: list[dict[str, str]] = []
    seen: dict[int, tuple[str, str]] = {}
    for path, name, port in ports:
        if not is_valid_port(port):
            continue
        if port in seen:
            previous_path, previous_name = seen[port]
            errors.append(
                {
                    "path": path,
                    "message": f"{name} conflicts with {previous_name}; both use local port {port}",
                }
            )
            errors.append(
                {
                    "path": previous_path,
                    "message": f"{previous_name} conflicts with {name}; both use local port {port}",
                }
            )
        else:
            seen[port] = (path, name)
    return errors


def validate_tensor_parallel_devices(config: VAPConfig) -> list[dict[str, str]]:
    tp_value = config.vllm_deploy_cfg.get("-tp")
    if tp_value is None:
        return []

    try:
        tensor_parallel_size = int(tp_value)
    except (TypeError, ValueError):
        return [
            {
                "path": "vllm_deploy_cfg.-tp",
                "message": "-tp must be a positive integer",
            }
        ]

    if tensor_parallel_size < 1:
        return [
            {
                "path": "vllm_deploy_cfg.-tp",
                "message": "-tp must be a positive integer",
            }
        ]

    devices = config.container_cfg.devices or []
    visible_device_count = len(devices) if devices else DEFAULT_VISIBLE_DEVICE_COUNT
    if tensor_parallel_size > visible_device_count:
        return [
            {
                "path": "vllm_deploy_cfg.-tp",
                "message": (
                    f"-tp={tensor_parallel_size} exceeds visible GPU device count "
                    f"{visible_device_count}. Empty devices means all "
                    f"{DEFAULT_VISIBLE_DEVICE_COUNT} GPUs are visible."
                ),
            }
        ]
    return []


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
    ensure_vap_home()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = TEMP_CONFIG_DIR / f"vap-config-{timestamp}-{uuid.uuid4().hex[:8]}.json"
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
        },
        {
            "name": "TensorBoard port",
            "port": config.profiler_cfg.tensorboard_port,
            "available": is_local_port_available(config.profiler_cfg.tensorboard_port),
        },
        {
            "name": "Perfetto Trace Processor port",
            "port": PERFETTO_PORT,
            "available": is_local_port_available(PERFETTO_PORT),
        },
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
        "message": (
            "Machine is reachable"
            if reachable
            else "Machine DNS resolved, but none of the checked ports are reachable"
        ),
    }


def resolve_config_path(raw_path: str | None) -> Path:
    if not raw_path:
        return CONFIG_PATH if CONFIG_PATH.is_file() else DEFAULT_CONFIG_PATH
    return resolve_under_vap_home(raw_path)


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
            "config_path": (
                str(RUN_STATE["config_path"]) if RUN_STATE["config_path"] else None
            ),
            "output": RUN_STATE["output"],
            "stop_requested": RUN_STATE["stop_requested"],
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


def build_log_download(file_name: str) -> tuple[str, bytes]:
    log_info = read_current_log_file(file_name)
    if not log_info.get("exists"):
        raise ValueError(str(log_info.get("message") or f"{file_name} does not exist"))
    return file_name, str(log_info["content"]).encode("utf-8")


def resolve_profile_archive_run_dir(raw_run_dir: str | None = None) -> Path:
    if raw_run_dir:
        run_dir = Path(raw_run_dir).resolve()
        if not run_dir.is_dir() or not run_dir.is_relative_to(LOGS_DIR.resolve()):
            raise ValueError("Invalid run directory for profile archive")
        return run_dir

    snapshot = get_run_state_snapshot()
    run_dir = Path(snapshot["run_dir"]).resolve() if snapshot["run_dir"] else None
    if run_dir is None:
        raise ValueError("There is no active or completed run directory yet")
    if not run_dir.is_dir() or not run_dir.is_relative_to(LOGS_DIR.resolve()):
        raise ValueError("Invalid current run directory for profile archive")
    return run_dir


def build_current_profile_archive(raw_run_dir: str | None = None) -> tuple[str, bytes]:
    run_dir = resolve_profile_archive_run_dir(raw_run_dir)
    profile_dir = (run_dir / "vllm-profile").resolve()
    if not profile_dir.is_dir() or not profile_dir.is_relative_to(LOGS_DIR.resolve()):
        raise ValueError("This run has not generated a vllm-profile directory yet")

    files = [path for path in profile_dir.rglob("*") if path.is_file()]
    if not files:
        raise ValueError("The vllm-profile directory is empty")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(run_dir))

    archive_name = f"{run_dir.name}-vllm-profile.zip"
    return archive_name, buffer.getvalue()


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


def monitor_run_process(
    process: subprocess.Popen[str], existing_dirs: set[str], started_at: float
) -> None:
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


def stop_process_group_sync(
    process: subprocess.Popen[str], timeout_sec: float = 5.0
) -> bool:
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
        is_running = (
            RUN_STATE["running"] and process is not None and process.poll() is None
        )
    if is_running:
        if not current_run_is_tensorboard_phase():
            raise RuntimeError("VAP is already running")
        with RUN_LOCK:
            RUN_STATE[
                "output"
            ] += "\n--- Previous TensorBoard is still running; stopping it before new run ---\n"
        if not stop_process_group_sync(process):
            raise RuntimeError(
                "The previous TensorBoard process did not stop in time. Try again later."
            )

    run_config_path = (config_path or resolve_config_path(None)).resolve()
    LOGS_DIR.mkdir(exist_ok=True)
    existing_dirs = {path.name for path in LOGS_DIR.iterdir() if path.is_dir()}
    started_at = time.time()
    process = subprocess.Popen(
        [sys.executable, "main.py", "run", "--config", str(run_config_path)],
        cwd=str(APP_DIR),
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
                "stop_requested": False,
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


def force_kill_process_group_later(
    process: subprocess.Popen[str], timeout_sec: float = 5.0
) -> None:
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


def process_cmdline(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except OSError:
        return ""


def terminate_recorded_visualization_pids(
    run_dir: Path | None, timeout_sec: float = 3.0
) -> None:
    if run_dir is None:
        return
    pid_file = (run_dir / "visualization_pids.json").resolve()
    if not pid_file.is_file() or not pid_file.is_relative_to(LOGS_DIR.resolve()):
        return
    try:
        payload = json.loads(pid_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to read visualization pid file {pid_file}: {exc}")
        return
    if not isinstance(payload, dict):
        return

    for name, raw_pid in payload.items():
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        cmdline = process_cmdline(pid)
        if not cmdline:
            continue
        if "tensorboard" not in cmdline and "trace_processor" not in cmdline:
            print(
                f"Skip cleanup for pid {pid}; command is not a VAP visualization process"
            )
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError as exc:
            print(f"Failed to terminate {name} pid {pid}: {exc}")
            continue

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                print(f"Failed to force kill {name} pid {pid}: {exc}")

    try:
        pid_file.write_text("{}\n", encoding="utf-8")
    except OSError:
        pass


def cleanup_active_run_on_server_exit(timeout_sec: float = 8.0) -> None:
    global SHUTDOWN_CLEANUP_DONE
    with SHUTDOWN_CLEANUP_LOCK:
        if SHUTDOWN_CLEANUP_DONE:
            return
        SHUTDOWN_CLEANUP_DONE = True

    with RUN_LOCK:
        process = RUN_STATE["process"]
        run_dir = Path(RUN_STATE["run_dir"]).resolve() if RUN_STATE["run_dir"] else None
        is_running = (
            RUN_STATE["running"] and process is not None and process.poll() is None
        )
        if is_running:
            RUN_STATE["stop_requested"] = True
            RUN_STATE[
                "output"
            ] += "\n--- Server is exiting; stopping active VAP run ---\n"

    if not is_running or process is None:
        terminate_recorded_visualization_pids(run_dir)
        return

    print("Stopping active VAP run before server exits...")
    if not stop_process_group_sync(process, timeout_sec=timeout_sec):
        print("Active VAP run did not stop cleanly before server exit.")
    terminate_recorded_visualization_pids(run_dir)


def stop_vap_run() -> dict[str, Any]:
    with RUN_LOCK:
        process = RUN_STATE["process"]
        run_dir = Path(RUN_STATE["run_dir"]).resolve() if RUN_STATE["run_dir"] else None
    if process is None or process.poll() is not None:
        terminate_recorded_visualization_pids(run_dir)
        return {"message": "There is no active VAP run", **get_run_state_snapshot()}
    terminate_run_process(process)
    force_kill_process_group_later(process)
    terminate_recorded_visualization_pids(run_dir)
    with RUN_LOCK:
        RUN_STATE["stop_requested"] = True
        RUN_STATE["output"] += "\n--- Stop requested from UI ---\n"
    return {
        "message": "Stop signal sent. main.py and its child processes will be stopped.",
        **get_run_state_snapshot(),
    }


def object_schema(
    properties: dict[str, Any] | None = None, required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


def current_config_payload() -> dict[str, Any]:
    config_path = resolve_config_path(None)
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def latest_log_run_dir() -> Path | None:
    if not LOGS_DIR.is_dir():
        return None
    candidates = [path for path in LOGS_DIR.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def inspect_latest_trace(args: dict[str, Any]) -> dict[str, Any]:
    preferred_name = str(args.get("preferred_name") or "merged_trace")
    raw_run_dir = args.get("run_dir")
    if isinstance(raw_run_dir, str) and raw_run_dir.strip():
        run_dir = resolve_profile_archive_run_dir(raw_run_dir)
    else:
        snapshot = get_run_state_snapshot()
        run_dir = (
            Path(snapshot["run_dir"]).resolve()
            if snapshot["run_dir"]
            else latest_log_run_dir()
        )
    if run_dir is None:
        raise ValueError("No run directory is available yet")

    profile_dir = (run_dir / "vllm-profile").resolve()
    if not profile_dir.is_dir() or not profile_dir.is_relative_to(LOGS_DIR.resolve()):
        raise ValueError(
            "The latest run has not generated a vllm-profile directory yet"
        )

    files = sorted([path for path in profile_dir.rglob("*") if path.is_file()])
    if not files:
        raise ValueError("The vllm-profile directory is empty")

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        preferred = preferred_name.lower()
        if name.startswith(preferred) or preferred in name:
            return (0, name)
        if "merged_trace" in name or "merge_trace" in name:
            return (1, name)
        if name.endswith(".trace.json.gz") or name.endswith(".pt.trace.json.gz"):
            return (2, name)
        if name.endswith(".trace.json") or name.endswith(".json"):
            return (3, name)
        return (4, name)

    trace_path = sorted(files, key=score)[0]
    stat = trace_path.stat()
    summary = summarize_trace_file(trace_path)
    return {
        "run_dir": str(run_dir),
        "profile_dir": str(profile_dir),
        "trace_path": str(trace_path),
        "trace_file": trace_path.name,
        "size_bytes": stat.st_size,
        "modified_at": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
        ),
        "candidate_count": len(files),
        "candidates": [path.name for path in sorted(files, key=score)[:10]],
        "looks_merged": trace_path.name.lower().startswith(
            ("merged_trace", "merge_trace")
        ),
        "summary": summary,
        "diagnosis": build_trace_diagnosis(summary),
    }


PERFETTO_SQL_QUERIES: dict[str, str] = {
    "trace_overview": """
SELECT
  (SELECT COUNT(*) FROM slice) AS slice_count,
  (SELECT COUNT(*) FROM thread_track) AS thread_track_count,
  (SELECT COUNT(*) FROM process) AS process_count,
  (SELECT COUNT(*) FROM sched) AS sched_count;
""",
    "top_slices": """
SELECT
  name,
  dur / 1000000.0 AS dur_ms,
  ts / 1000000.0 AS ts_ms
FROM slice
WHERE dur > 0
ORDER BY dur DESC
LIMIT {limit};
""",
    "category_duration": """
SELECT
  COALESCE(category, 'uncategorized') AS category,
  COUNT(*) AS event_count,
  SUM(dur) / 1000000.0 AS total_dur_ms,
  AVG(dur) / 1000000.0 AS avg_dur_ms,
  MAX(dur) / 1000000.0 AS max_dur_ms
FROM slice
WHERE dur > 0
GROUP BY category
ORDER BY total_dur_ms DESC
LIMIT {limit};
""",
    "sync_events": """
SELECT
  name,
  dur / 1000000.0 AS dur_ms,
  ts / 1000000.0 AS ts_ms
FROM slice
WHERE dur > 0
  AND (
    name LIKE '%Synchronize%'
    OR name LIKE '%sync%'
    OR name LIKE '%Wait%'
    OR name LIKE '%wait%'
    OR name LIKE '%barrier%'
  )
ORDER BY dur DESC
LIMIT {limit};
""",
    "gpu_kernels": """
SELECT
  name,
  dur / 1000000.0 AS dur_ms,
  ts / 1000000.0 AS ts_ms
FROM slice
WHERE dur > 0
  AND (
    category LIKE '%kernel%'
    OR name LIKE '%Kernel%'
    OR name LIKE '%hipLaunchKernel%'
  )
ORDER BY dur DESC
LIMIT {limit};
""",
    "operator_hotspots": """
SELECT
  name,
  COUNT(*) AS calls,
  SUM(dur) / 1000000.0 AS total_dur_ms,
  AVG(dur) / 1000000.0 AS avg_dur_ms,
  MAX(dur) / 1000000.0 AS max_dur_ms
FROM slice
WHERE dur > 0
  AND (
    name LIKE 'aten::%'
    OR name LIKE 'vllm%'
    OR name LIKE '%attention%'
    OR name LIKE '%Attention%'
  )
GROUP BY name
ORDER BY total_dur_ms DESC
LIMIT {limit};
""",
    "timeline_gaps": """
SELECT
  name,
  dur / 1000000.0 AS dur_ms,
  ts / 1000000.0 AS ts_ms
FROM slice
WHERE dur > 0
  AND name LIKE '%idle%'
ORDER BY dur DESC
LIMIT {limit};
""",
}


def load_skill_queries(skill_dir: Path = TORCHPROFILER_SKILL_DIR) -> dict[str, str]:
    query_file = skill_dir / "queries.yaml"
    if not query_file.is_file():
        return PERFETTO_SQL_QUERIES
    queries: dict[str, list[str]] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for raw_line in query_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" ") and raw_line.endswith(": |"):
            if current_name is not None:
                queries[current_name] = current_lines
            current_name = raw_line.split(":", 1)[0].strip()
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(
                raw_line[2:] if raw_line.startswith("  ") else raw_line
            )
    if current_name is not None:
        queries[current_name] = current_lines
    parsed = {name: "\n".join(lines).strip() for name, lines in queries.items()}
    return parsed or PERFETTO_SQL_QUERIES


def torchprofiler_skill_workflows() -> dict[str, list[str]]:
    return {
        "overview": ["trace_overview", "category_duration", "rank_activity"],
        "sync_waits": ["sync_waits", "top_slices"],
        "operator_hotspots": ["operator_hotspots", "aten_hotspots"],
        "gpu_kernels": ["gpu_kernels", "category_duration"],
        "rank_imbalance": ["rank_activity", "rank_longest_slices"],
        "memory_copy": ["memory_copies", "category_duration"],
        "full_report": [
            "trace_overview",
            "category_duration",
            "sync_waits",
            "gpu_kernels",
            "operator_hotspots",
            "rank_activity",
            "memory_copies",
        ],
    }


def trace_processor_path() -> Path:
    candidates = [
        VAP_BIN_DIR / "trace_processor",
        APP_DIR / "bin" / "trace_processor",
        APP_DIR / "trace_processor",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise ValueError("trace_processor is not installed. Run install.sh first.")


def run_perfetto_sql(args: dict[str, Any]) -> dict[str, Any]:
    queries = load_skill_queries()
    query_name = str(args.get("query_name") or "top_slices")
    if query_name not in queries:
        raise ValueError(
            "Unsupported query_name. Use one of: " + ", ".join(sorted(queries))
        )
    limit = args.get("limit", 20)
    if not isinstance(limit, int) or limit < 1 or limit > 100:
        raise ValueError("limit must be an integer from 1 to 100")

    trace_info = inspect_latest_trace(
        {
            "preferred_name": args.get("preferred_name") or "merged_trace",
            "run_dir": args.get("run_dir"),
        }
    )
    trace_path = Path(trace_info["trace_path"])
    sql = queries[query_name].format(limit=limit)
    perfetto_home = VAP_PERFETTO_HOME
    perfetto_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(perfetto_home)
    command = [str(trace_processor_path()), "query", str(trace_path), sql]
    try:
        completed = subprocess.run(
            command,
            cwd=str(APP_DIR),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Perfetto SQL query timed out after 120s") from exc
    if completed.returncode != 0:
        raise ValueError(
            "Perfetto SQL query failed: "
            + (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"exit {completed.returncode}"
            )
        )
    return {
        "query_name": query_name,
        "sql": sql.strip(),
        "trace_file": trace_info["trace_file"],
        "trace_path": trace_info["trace_path"],
        "looks_merged": trace_info.get("looks_merged"),
        "stdout": completed.stdout.strip()[:12000],
        "stderr": completed.stderr.strip()[:4000],
    }


def run_torchprofiler_skill(args: dict[str, Any]) -> dict[str, Any]:
    workflow = str(args.get("workflow") or "full_report")
    workflows = torchprofiler_skill_workflows()
    if workflow not in workflows:
        raise ValueError(
            "Unsupported workflow. Use one of: " + ", ".join(sorted(workflows))
        )
    limit = args.get("limit", 20)
    results = []
    for query_name in workflows[workflow]:
        try:
            results.append(
                {
                    "query_name": query_name,
                    "ok": True,
                    "result": run_perfetto_sql(
                        {
                            "query_name": query_name,
                            "preferred_name": args.get("preferred_name")
                            or "merged_trace",
                            "run_dir": args.get("run_dir"),
                            "limit": limit,
                        }
                    ),
                }
            )
        except Exception as exc:
            results.append({"query_name": query_name, "ok": False, "message": str(exc)})
    return {
        "skill": "TorchProfilerTraceSkill",
        "workflow": workflow,
        "attribution": "Inspired by Gracker/Perfetto-Skills evidence-driven workflow design; original VAP SQL presets.",
        "results": results,
    }


def build_trace_diagnosis(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary.get("available"):
        return {
            "available": False,
            "findings": [summary.get("message", "Trace summary is unavailable")],
        }

    findings: list[str] = []
    next_steps: list[str] = []
    hypotheses: list[str] = []
    duration_categories = {
        item["category"]: item["duration_us"]
        for item in summary.get("top_categories_by_duration_us", [])
        if isinstance(item, dict)
    }
    longest = summary.get("longest_events", [])
    longest_names = [
        str(item.get("name", "")) for item in longest if isinstance(item, dict)
    ]

    if (
        duration_categories.get("cuda_runtime", 0)
        > duration_categories.get("kernel", 0) * 10
    ):
        findings.append(
            "cuda_runtime duration is much larger than kernel duration; synchronization or launch overhead may dominate the trace."
        )
        hypotheses.append(
            "Inspect hipEventSynchronize and hipLaunchKernel spans for blocking waits, serialized work, or host-side scheduling gaps."
        )
    if any("hipEventSynchronize" in name for name in longest_names):
        findings.append(
            "The longest events include hipEventSynchronize, which often points to host waiting for GPU completion or synchronization barriers."
        )
        next_steps.append(
            "In Perfetto, focus on hipEventSynchronize regions and check what GPU work precedes each wait."
        )
    if any("aten::sort" in name for name in longest_names):
        findings.append(
            "The longest CPU ops include aten::sort across ranks; sampling/top-k/sorting work may be a decode bottleneck."
        )
        hypotheses.append(
            "Review sampling settings and decode path; compare whether aten::sort aligns with token generation steps."
        )

    ranks = [
        (rank, count)
        for rank, count in summary.get("events_by_rank", [])
        if rank != "unknown"
    ]
    if ranks:
        counts = [count for _, count in ranks]
        if max(counts) - min(counts) > max(counts) * 0.10:
            findings.append(
                "Event counts differ noticeably across ranks; possible rank imbalance."
            )
        else:
            findings.append("Rank event counts look roughly balanced.")
        next_steps.append(
            "In Perfetto, compare rank lanes for idle gaps, long waits, and whether decode steps align across ranks."
        )

    next_steps.extend(
        [
            "Inspect user_annotation events to separate prefill/decode phases and request-level spans.",
            "Use TensorBoard profiler views to cross-check operator time, kernel time, memory copies, and trace step boundaries.",
            "Compare GPU kernel lanes with CPU op lanes to identify host gaps before GPU work launches.",
        ]
    )

    return {
        "available": True,
        "findings": findings[:8],
        "next_steps": next_steps[:8],
        "optimization_hypotheses": hypotheses[:8],
    }


def summarize_trace_file(trace_path: Path) -> dict[str, Any]:
    try:
        if trace_path.suffix == ".gz":
            with gzip.open(
                trace_path, "rt", encoding="utf-8", errors="replace"
            ) as trace_file:
                payload = json.load(trace_file)
        else:
            payload = json.loads(
                trace_path.read_text(encoding="utf-8", errors="replace")
            )
    except Exception as exc:
        return {"available": False, "message": f"Failed to parse trace JSON: {exc}"}

    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return {
            "available": False,
            "message": "Trace JSON does not contain traceEvents",
        }

    category_counts: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    duration_by_category: defaultdict[str, float] = defaultdict(float)
    longest_events: list[dict[str, Any]] = []
    min_ts: float | None = None
    max_ts: float | None = None

    for event in events:
        if not isinstance(event, dict):
            continue
        category = str(event.get("cat") or "uncategorized")
        category_counts[category] += 1
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        rank = args.get("rank", "unknown")
        rank_counts[str(rank)] += 1

        ts = event.get("ts")
        dur = event.get("dur")
        if isinstance(ts, (int, float)):
            min_ts = ts if min_ts is None else min(min_ts, ts)
            if isinstance(dur, (int, float)):
                max_ts = ts + dur if max_ts is None else max(max_ts, ts + dur)
            else:
                max_ts = ts if max_ts is None else max(max_ts, ts)
        if isinstance(dur, (int, float)):
            duration_by_category[category] += float(dur)
            longest_events.append(
                {
                    "name": str(event.get("name") or ""),
                    "category": category,
                    "duration_us": round(float(dur), 3),
                    "rank": str(rank),
                    "pid": event.get("pid"),
                    "tid": event.get("tid"),
                }
            )

    longest_events = sorted(
        longest_events,
        key=lambda item: item["duration_us"],
        reverse=True,
    )[:20]

    return {
        "available": True,
        "event_count": len(events),
        "time_span_us": (
            round(max_ts - min_ts, 3)
            if min_ts is not None and max_ts is not None
            else None
        ),
        "top_categories_by_count": category_counts.most_common(12),
        "top_categories_by_duration_us": sorted(
            (
                {"category": category, "duration_us": round(duration, 3)}
                for category, duration in duration_by_category.items()
            ),
            key=lambda item: item["duration_us"],
            reverse=True,
        )[:12],
        "events_by_rank": rank_counts.most_common(),
        "longest_events": longest_events,
    }


def prepare_download_artifact(args: dict[str, Any]) -> dict[str, Any]:
    artifact = str(args.get("artifact") or "")
    log_names = {
        "vap_log": "vap_log.txt",
        "vllm_deploy_log": "vllm_deploy.log",
        "vllm_bench_log": "vllm_bench.log",
    }
    if artifact in log_names:
        file_name = log_names[artifact]
        log_info = read_current_log_file(file_name)
        if not log_info.get("exists"):
            raise ValueError(
                str(log_info.get("message") or f"{file_name} does not exist")
            )
        return {
            "artifact": artifact,
            "label": file_name,
            "download_url": f"/api/log-file/download?name={file_name}",
            "content_type": "text/plain",
        }
    if artifact == "trace_archive":
        run_dir = get_run_state_snapshot().get("run_dir")
        # Validate availability now, but let the browser stream the real download.
        file_name, _ = build_current_profile_archive(str(run_dir) if run_dir else None)
        query = f"?run_dir={str(run_dir)}" if run_dir else ""
        return {
            "artifact": artifact,
            "label": file_name,
            "download_url": f"/api/profile/archive{query}",
            "content_type": "application/zip",
        }
    raise ValueError(
        "Unsupported artifact. Use vap_log, vllm_deploy_log, vllm_bench_log, or trace_archive."
    )


def get_agent_runtime() -> VAPAgentRuntime:
    global AGENT_TOOLS_REGISTERED
    with AGENT_TOOLS_LOCK:
        if not AGENT_TOOLS_REGISTERED:
            register_vap_agent_tools(AGENT_RUNTIME)
            AGENT_TOOLS_REGISTERED = True
    return AGENT_RUNTIME


def get_agent_status_payload() -> dict[str, Any]:
    return {
        **get_agent_runtime().status(),
        "server_session_id": SERVER_SESSION_ID,
    }


def register_vap_agent_tools(runtime: VAPAgentRuntime) -> None:
    runtime.register_tool(
        AgentTool(
            name="get_config",
            description="Read the saved VAP config payload.",
            safety="read_only",
            parameters=object_schema(),
            handler=lambda args: {"config": current_config_payload()},
        )
    )
    runtime.register_tool(
        AgentTool(
            name="get_run_status",
            description="Read current VAP run status without changing any process.",
            safety="read_only",
            parameters=object_schema(),
            handler=lambda args: get_run_state_snapshot(),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="read_log_file",
            description="Read one current run log file.",
            safety="read_only",
            parameters=object_schema(
                {
                    "file_name": {
                        "type": "string",
                        "enum": ["vap_log.txt", "vllm_deploy.log", "vllm_bench.log"],
                    }
                },
                ["file_name"],
            ),
            handler=lambda args: read_current_log_file(str(args["file_name"])),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="validate_config",
            description="Validate a VAP config payload. If config is omitted, validate the saved config.",
            safety="safe",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=lambda args: validate_config_payload(
                args.get("config") or current_config_payload()
            ),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="check_ports",
            description="Check local VAP service ports. If config is omitted, use the saved config.",
            safety="safe",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=lambda args: check_config_ports(
                args.get("config") or current_config_payload()
            ),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="check_resources",
            description="Check model paths, Docker image, devices, and mount sources.",
            safety="safe",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=lambda args: check_config_resources(
                args.get("config") or current_config_payload()
            ),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="inspect_latest_trace",
            description="Inspect the latest profiling trace metadata and a small preview. Defaults to merged_trace.",
            safety="read_only",
            parameters=object_schema(
                {
                    "preferred_name": {
                        "type": "string",
                        "description": "Preferred trace filename prefix. Defaults to merged_trace.",
                    },
                    "run_dir": {
                        "type": "string",
                        "description": "Optional run directory under logs.",
                    },
                }
            ),
            handler=inspect_latest_trace,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="run_perfetto_sql",
            description="Run a whitelisted Perfetto SQL query on the latest trace. Use this for detailed trace analysis without loading raw JSON into the LLM.",
            safety="read_only",
            parameters=object_schema(
                {
                    "query_name": {
                        "type": "string",
                        "enum": sorted(PERFETTO_SQL_QUERIES),
                    },
                    "preferred_name": {
                        "type": "string",
                        "description": "Preferred trace filename prefix. Defaults to merged_trace.",
                    },
                    "run_dir": {
                        "type": "string",
                        "description": "Optional run directory under logs.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                ["query_name"],
            ),
            handler=run_perfetto_sql,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="run_torchprofiler_skill",
            description="Run a TorchProfilerTraceSkill workflow made of VAP-owned Perfetto SQL presets.",
            safety="read_only",
            parameters=object_schema(
                {
                    "workflow": {
                        "type": "string",
                        "enum": sorted(torchprofiler_skill_workflows()),
                    },
                    "preferred_name": {
                        "type": "string",
                        "description": "Preferred trace filename prefix. Defaults to merged_trace.",
                    },
                    "run_dir": {
                        "type": "string",
                        "description": "Optional run directory under logs.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                ["workflow"],
            ),
            handler=run_torchprofiler_skill,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="prepare_download_artifact",
            description="Prepare a safe download link for current run logs or the trace archive.",
            safety="safe",
            parameters=object_schema(
                {
                    "artifact": {
                        "type": "string",
                        "enum": [
                            "vap_log",
                            "vllm_deploy_log",
                            "vllm_bench_log",
                            "trace_archive",
                        ],
                    }
                },
                ["artifact"],
            ),
            handler=prepare_download_artifact,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="prepare_run",
            description="Validate whether a config is ready to run without starting VAP.",
            safety="safe",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=lambda args: validate_config_payload(
                args.get("config") or current_config_payload()
            ),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="start_run",
            description="Start a VAP run. Requires explicit user approval.",
            safety="requires_approval",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=lambda args: start_vap_run(
                save_temp_config(args["config"])
                if isinstance(args.get("config"), dict)
                else None
            ),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="stop_run",
            description="Stop the active VAP run. Requires explicit user approval.",
            safety="requires_approval",
            parameters=object_schema(),
            handler=lambda args: stop_vap_run(),
        )
    )


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
        if parsed.path == "/api/log-file/download":
            self.handle_log_download(parsed.query)
            return
        if parsed.path == "/api/run/status":
            self.handle_run_status()
            return
        if parsed.path == "/api/profile/archive":
            self.handle_profile_archive(parsed.query)
            return
        if parsed.path == "/api/agent/status":
            self.handle_agent_status()
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
            "/api/agent/unlock": self.handle_agent_unlock,
            "/api/agent/chat": self.handle_agent_chat,
            "/api/agent/chat/stream": self.handle_agent_chat_stream,
            "/api/agent/approve": self.handle_agent_approve,
            "/api/agent/cancel-action": self.handle_agent_cancel_action,
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
        except TimeoutError as exc:
            self.send_json(
                {"message": f"Request timed out: {exc}"},
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        except Exception as exc:
            if "timed out" in str(exc).lower():
                self.send_json(
                    {"message": f"Request timed out: {exc}"},
                    HTTPStatus.GATEWAY_TIMEOUT,
                )
                return
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

    def handle_log_download(self, query: str) -> None:
        try:
            params = parse_qs(query)
            file_name = params.get("name", [""])[0]
            download_name, content = build_log_download(file_name)
        except Exception as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_binary(
            content,
            "text/plain; charset=utf-8",
            f'attachment; filename="{download_name}"',
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

    def handle_profile_archive(self, query: str) -> None:
        try:
            params = parse_qs(query)
            run_dir = params.get("run_dir", [None])[0]
            file_name, content = build_current_profile_archive(run_dir)
        except ValueError as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_binary(
            content,
            "application/zip",
            f'attachment; filename="{file_name}"',
        )

    def handle_validate(self) -> None:
        payload = self.read_json_body()
        self.send_json(validate_config_payload(payload))

    def handle_agent_status(self) -> None:
        self.send_json(get_agent_status_payload())

    def handle_agent_unlock(self) -> None:
        payload = self.read_json_body()
        subscription_key = payload.get("subscription_key")
        if not isinstance(subscription_key, str):
            raise ValueError("subscription_key is required")
        self.send_json(
            {
                **get_agent_runtime().unlock(subscription_key),
                "server_session_id": SERVER_SESSION_ID,
            }
        )

    def handle_agent_chat(self) -> None:
        payload = self.read_json_body()
        self.send_json(get_agent_runtime().chat(payload))

    def handle_agent_chat_stream(self) -> None:
        payload = self.read_json_body()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send_event(event: dict[str, Any]) -> None:
            content = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode(
                "utf-8"
            )
            self.wfile.write(content)
            self.wfile.flush()

        try:
            for event in get_agent_runtime().stream_chat(payload):
                send_event(event)
        except Exception as exc:
            send_event({"type": "error", "message": str(exc)})

    def handle_agent_approve(self) -> None:
        payload = self.read_json_body()
        approval_id = payload.get("approval_id")
        if not isinstance(approval_id, str):
            raise ValueError("approval_id is required")
        self.send_json(get_agent_runtime().approve(approval_id))

    def handle_agent_cancel_action(self) -> None:
        payload = self.read_json_body()
        approval_id = payload.get("approval_id")
        if not isinstance(approval_id, str):
            raise ValueError("approval_id is required")
        self.send_json(get_agent_runtime().cancel(approval_id))

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

    def send_binary(
        self,
        content: bytes,
        content_type: str,
        content_disposition: str | None = None,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        if content_disposition:
            self.send_header("Content-Disposition", content_disposition)
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[VAP Config UI] {self.address_string()} - {fmt % args}")


def main(argv: list[str] | None = None) -> None:
    ensure_vap_home()
    parser = argparse.ArgumentParser(description="VAP config management UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args(argv)

    atexit.register(cleanup_active_run_on_server_exit)

    def handle_shutdown_signal(signum: int, frame: Any) -> None:
        print(f"Received signal {signum}; shutting down VAP config UI...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    server = ThreadingHTTPServer((args.host, args.port), VAPConfigHandler)
    print(f"VAP config UI started: http://{args.host}:{args.port}")
    print(f"VAP home: {VAP_HOME}")
    print(f"Temporary config files will be saved to: {TEMP_CONFIG_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("VAP config UI is stopping...")
    finally:
        cleanup_active_run_on_server_exit()
        server.server_close()


if __name__ == "__main__":
    main()
