import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import torch
from diffusers import (
    DiffusionPipeline,  # type: ignore
    DPMSolverMultistepScheduler,  # type: ignore
    DPMSolverSinglestepScheduler,  # type: ignore
    StableDiffusionXLPipeline,  # type: ignore
)

# from PIL import Image as PILImage
from PIL.Image import Image

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
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

    TODO:
    - What is the best wat to cache the model? lru_cache or global parameter?

    """
    MODELS_ROOT_PATH = os.environ.get("MODELS_ROOT_PATH")

    if MODELS_ROOT_PATH is None:
        raise ValueError("MODELS_ROOT_PATH is not set")

    model_path = Path(MODELS_ROOT_PATH) / "dreamshaperXL_turboDpmppSDE.safetensors"
    image_size = 1024

    pipe = StableDiffusionXLPipeline.from_single_file(
        pretrained_model_link_or_path=str(model_path.resolve()),
        image_size=image_size,
        torch_dtype=torch.bfloat16,
        variant="fp16",
        # vae=None,
        # local_files_only=True,
    )

    # if allow_nsfw is True:
    #     pipe.safety_checker = None

    # Default scheduler: EulerDiscreteScheduler

    # NOTE: Under dev, Related PR: https://github.com/huggingface/diffusers/pull/6477/files
    if scheduler == "DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            config=pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_order=2,
            final_sigmas_type="zero",
        )
    elif scheduler == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )

    # pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    return pipe


def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
    # adjust the batch_size of prompt_embeds according to guidance_scale
    if step_index == int(pipe.num_timestep * 0.4):
        prompt_embeds = callback_kwargs["prompt_embeds"]
        prompt_embeds = prompt_embeds.chunk(2)[-1]

    # update guidance_scale and prompt_embeds
    pipe._guidance_scale = 0.0
    callback_kwargs["prompt_embeds"] = prompt_embeds
    return callback_kwargs


#### PROMPT FUSION START ####
def create_prompt_embeds(pipe, prompt):
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device=pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
    )
    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds


def callback_prompt_fusion(pipe, step_index, timestep, callback_kwargs):
    # NOTE: Should be global
    prompt_update_schedule = {8: "bird", 11: "girl"}

    if step_index in prompt_update_schedule:
        prompt = prompt_update_schedule[step_index]
        prompt_embeds = create_prompt_embeds(pipe, prompt)
        callback_kwargs["prompt_embeds"] = prompt_embeds

    return callback_kwargs


#### PROMPT FUSION END ####


#### INTERMATIATE IMAGES START ####


def callback_image_progress(pipe, step_index, timestep, callback_kwargs):
    """Callback for getting intermediate images for stream previewing..

    Args:
        pipe: _description_.
        step_index: _description_.
        timestep: _description_.
        callback_kwargs: _description_.

    NOTE:
    - Doesn't work if scheduler is `DPM++ SDE Karras`.

    Returns:
        _description_.
    """

    if step_index % 2 != 0:
        return callback_kwargs

    latents = callback_kwargs["latents"]

    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample

        image = (image * 0.5 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # convert to PIL Images
        image = pipe.numpy_to_pil(image)

        # do something with the Images
        for i, img in enumerate(image):
            img.save(f"step_{step_index}_img{i}.png")

    return callback_kwargs


#### INTERMATIATE IMAGES END ####


def text_to_img(  # noqa: PLR0913
    prompt: str,
    negative_prompt: str | None = None,
    seed: int | None = None,
    num_inference_steps: int = 6,
    cfg_scale: float = 2.5,
    clip_skip: int | None = None,
    image_size: int = 1024,
) -> Image:
    generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)

    # pipe = initialize_model(scheduler=None)
    # pipe = initialize_model(scheduler="DPM++ SDE Karras")

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
        # callback_on_step_end=callback_image_progress,
        # callback_on_step_end_tensor_inputs=[
        #     "latents",
        #     # "prompt_embeds",
        #     # "negative_prompt_embeds",
        # ],
    ).images[0]  # type: ignore

    end_time = time.time()

    logger.info(f"Time taken: {end_time - start_time}")

    return image


def img_to_img(  # noqa: PLR0913
    prompt: str,
    negative_prompt: str | None = None,
    seed: int | None = None,
    num_inference_steps: int = 6,
    cfg_scale: float = 2.5,
    clip_skip: int | None = None,
    image_size: int = 1024,
):
    logger.info(f"Using device: {pipe.device}, with seed: {seed}")
    # start_time = time.time()


#############################################################################

# scheduler = None
scheduler = "DPM++ SDE Karras"
# scheduler = "DPM++ 2M Karras"

pipe = initialize_model(scheduler=scheduler)
