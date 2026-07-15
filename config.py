from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import os


class ModelConfig(BaseModel):
    model_name: str
    model_path: str


class DistributedConfig(BaseModel):
    num_nodes: int
    ray_port: int
    head_node: str
    worker_nodes: List[str]


class ProfilerConfig(BaseModel):
    profiler: str
    torch_profiler_dir: str
    torch_profiler_record_shapes: bool
    torch_profiler_with_stack: bool
    torch_profiler_with_memory: bool
    torch_profiler_with_flops: bool
    torch_profiler_use_gzip: bool
    delay_iterations: int = 0
    max_iterations: int = 0


class MountConfig(BaseModel):
    target: str
    source: str
    type: Optional[str] = "bind"


class DockerConfig(BaseModel):
    image_name: str
    image_tag: str
    devices: Optional[List[str]] = None
    mounts: Optional[List[MountConfig]] = None
    env_vars: Optional[Dict[str, str]] = None


class VAPConfig(BaseModel):
    model_cfg: ModelConfig
    distributed_cfg: Optional[DistributedConfig] = None
    vllm_deploy_cfg: Dict[str, Any]
    vllm_bench_cfg: Dict[str, Any]
    profiler_cfg: ProfilerConfig
    container_cfg: DockerConfig

    @property
    def docker_image(self) -> str:
        return f"{self.container_cfg.image_name}:{self.container_cfg.image_tag}"

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_cfg.model_path, self.model_cfg.model_name)

    @property
    def vllm_host(self) -> str:
        assert "--host" in self.vllm_deploy_cfg
        assert "--host" in self.vllm_bench_cfg
        assert self.vllm_deploy_cfg["--host"] == self.vllm_bench_cfg["--host"]
        return self.vllm_deploy_cfg["--host"]

    @property
    def vllm_port(self) -> int:
        assert "--port" in self.vllm_deploy_cfg
        assert "--port" in self.vllm_bench_cfg
        assert self.vllm_deploy_cfg["--port"] == self.vllm_bench_cfg["--port"]
        return self.vllm_deploy_cfg["--port"]

    def build_profiler_cli_args_dict(self) -> dict[str, object]:
        args: dict[str, object] = {}
        for k, v in self.profiler_cfg.model_dump().items():
            if isinstance(v, bool):
                v = str(v).lower()
            args[f"--profiler-config.{k}"] = v
        return args

    def vllm_deploy_args_str(self) -> str:

        deploy_args = dict(self.vllm_deploy_cfg)
        deploy_args.update(self.build_profiler_cli_args_dict())
        return self.build_cli_rgs_str(deploy_args)

    def vllm_bench_args_str(self) -> str:
        bench_args = dict(self.vllm_bench_cfg)
        return self.build_cli_rgs_str(bench_args)

    def build_cli_rgs_str(self, args_dict: Dict[str, Any]) -> str:
        return " ".join(
            f"{k} {v}" if v is not None else k for k, v in args_dict.items()
        )
