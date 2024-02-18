import gc
import os

import torch
from huggingface_hub import snapshot_download


def check_env_vars(env_vars: list[str] | None = None) -> None:
    """
    Checks if the required environment variables are set.

    Args:
        env_vars (list[str], optional): List of environment variables to check. Defaults to None.

    Raises:
        ValueError: If any of the environment variables are not set.
    """
    if env_vars is None:
        env_vars = ["LIBRARY_BASE_PATH", "MODELS_ROOT_PATH"]

    for env_var in env_vars:
        if os.getenv(env_var) is None:
            raise ValueError(f"Please set {env_var} env var.")


def download_model_from_hf(
    repo_id: str, sub_dir: str | None, revision: str = "main"
) -> None:
    check_env_vars(env_vars=["MODELS_ROOT_PATH"])

    MODELS_ROOT_PATH = os.environ["MODELS_ROOT_PATH"]

    local_dir_name = repo_id.split("/")[1]

    if sub_dir is not None:
        local_dir_name = f"{sub_dir}/{local_dir_name}"

    # NOTE: First downloads to cache and then copies to local_dir
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=f"{MODELS_ROOT_PATH}/{local_dir_name}",
        local_dir_use_symlinks=False,
        ignore_patterns="*.pt",
    )


def flush_gpu_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    repo_id = "stabilityai/sdxl-turbo"
    sub_dir = None

    download_model_from_hf(repo_id=repo_id, sub_dir=sub_dir)
