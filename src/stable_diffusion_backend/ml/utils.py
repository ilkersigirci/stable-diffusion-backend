import os

from huggingface_hub import snapshot_download

MODELS_ROOT_PATH = os.environ.get("MODELS_ROOT_PATH", None)

if MODELS_ROOT_PATH is None:
    raise ValueError("MODELS_ROOT_PATH is not set")


def download_model_from_hf(
    repo_id: str, sub_dir: str | None, revision: str = "main"
) -> None:
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


if __name__ == "__main__":
    repo_id = "stabilityai/sdxl-turbo"
    sub_dir = None

    download_model_from_hf(repo_id=repo_id, sub_dir=sub_dir)
