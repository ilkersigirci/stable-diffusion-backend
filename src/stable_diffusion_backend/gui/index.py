import os

import streamlit as st

LIBRARY_BASE_PATH = os.environ.get("LIBRARY_BASE_PATH")

st.set_page_config(
    page_title="Stable-Diffusion-Backend",
    page_icon="👋",
)
st.write("# :blue[Stable-Diffusion-Backend] on the web !")


# st.info("For more information please visit our website")
st.sidebar.success("Select a page")
