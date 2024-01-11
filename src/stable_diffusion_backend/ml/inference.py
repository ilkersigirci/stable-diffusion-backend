import logging
import time
from pathlib import Path

import torch
from diffusers import (
    DiffusionPipeline,  # type: ignore
    DPMSolverSinglestepScheduler,  # type: ignore
    StableDiffusionXLPipeline,  # type: ignore
)
from PIL.Image import Image

logger = logging.getLogger(__name__)


def initialize_model(
    scheduler: str | None = None, allow_nsfw: bool = True
) -> DiffusionPipeline:
    """
    Initialize the model for inference.

    NOTE:
    - Fast optimization: https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion
    - VRAM optimizations: https://huggingface.co/docs/diffusers/optimization/memory

    - `enable_model_cpu_offload()` is the only option to handle 8 GB VRAM for SDXL models.
        - `enable_vae_slicing()`, `enable_vae_tiling()`, `enable_sequential_cpu_offload()` are not needed.

    """
    model_path = (
        Path(__file__).parent / "models/dreamshaperXL_turboDpmppSDE.safetensors"
    )
    image_size = 1024

    pipe = StableDiffusionXLPipeline.from_single_file(
        pretrained_model_link_or_path=str(model_path.resolve()),
        image_size=image_size,
        torch_dtype=torch.bfloat16,
        vae=None,
        # local_files_only=True,
    )

    if allow_nsfw is True:
        pipe.safety_checker = None

    # Default scheduler: EulerDiscreteScheduler
    if scheduler == "DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )

    # pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    return pipe


pipe = initialize_model(scheduler="DPM++ SDE Karras")


def generate_image(  # noqa: PLR0913
    prompt: str,
    negative_prompt: str | None = None,
    seed: int | None = None,
    num_inference_steps: int = 6,
    cfg_scale: float = 2.5,
    clip_skip: int | None = None,
    image_size: int = 1024,
) -> Image:
    generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)

    logger.info(f"Using device: {pipe.device}, with seed: {seed}")

    start_time = time.time()

    image: Image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=cfg_scale,  # (CFG: Classifier-Free Diffusion Guidance)
        num_inference_steps=num_inference_steps,
        generator=generator,
        clip_skip=clip_skip,
        width=image_size,
        height=image_size,
    ).images[0]  # type: ignore

    end_time = time.time()

    logger.info(f"Time taken: {end_time - start_time}")

    return image
