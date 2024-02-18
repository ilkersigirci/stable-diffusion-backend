import streamlit as st
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from stable_diffusion_backend.ml.image_generation import initialize_model, text_to_img
from stable_diffusion_backend.utils.streamlit_ui import streamlit_default_label


def load_model(scheduler: str) -> DiffusionPipeline:
    return initialize_model(scheduler=scheduler)


def main() -> None:
    schedulers = ["DPM++ SDE Karras", "EulerDiscreteScheduler", "DPM++ 2M Karras"]

    scheduler = streamlit_default_label(
        label="Choose a scheduler", options=schedulers, widget="selectbox"
    )

    if not scheduler:
        st.stop()

    with st.spinner("Loading model..."):
        model = load_model(scheduler=scheduler)

    prompt = st.text_input("Prompt", None)
    negative_prompt = st.text_input("Negative Prompt", None)
    seed = st.number_input("Seed", min_value=0, max_value=100_000, value=None)
    num_inference_steps = st.number_input(
        "Inference Steps", min_value=0, max_value=20, value=7
    )
    cfg_scale = st.number_input("Cfg Scale", 3.0)
    clip_skip = st.number_input("Skip clip", 2)

    generate_button = st.button("Generate Image", None)

    if generate_button is None:
        st.stop()

    if generate_button is not None and prompt is None:
        st.warning("Please enter a prompt")
        st.stop()

    with st.spinner("Generating image..."):
        # TODO: Add callback to inference step to show it as a progress bar
        image = text_to_img(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            clip_skip=clip_skip,
        )

        st.image(image)


if __name__ == "__main__":
    main()
