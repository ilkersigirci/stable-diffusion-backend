import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from PIL.Image import Image

# from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline

logger = logging.getLogger(__name__)
model_path = Path(__file__).parent / "models/dreamshaperXL_turboDpmppSDE.safetensors"
image_size = 1024
file_path = Path(__file__).parent


pipe = StableDiffusionXLPipeline.from_single_file(
    pretrained_model_link_or_path=model_path.resolve(),
    image_size=image_size,
    torch_dtype=torch.float16,
    vae=None,
    # local_files_only=True,
)

# pipe.to("cuda")

# NOTE: VRAM Optimizations
# https://huggingface.co/docs/diffusers/optimization/memory

# pipe.enable_vae_slicing()
# pipe.enable_vae_tiling()

# pipe.enable_sequential_cpu_offload()

# NOTE: The best option for reducing VRAM usage.
pipe.enable_model_cpu_offload()


def obtain_image(
    prompt: str,
    seed: int | None = None,
    num_inference_steps: int = 6,
    guidance_scale: float = 2.5,
) -> Image:
    generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)

    logger.info(f"Using device: {pipe.device}, with seed: {seed}")

    image: Image = pipe(
        [prompt],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        width=image_size,
        height=image_size,
    ).images[0]

    return image
