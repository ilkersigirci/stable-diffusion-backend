from typing import Any, List

import streamlit as st


def streamlit_default_label(
    label: str,
    options: List[Any],
    widget: str = "selectbox",
):
    widget_mapper = {
        "selectbox": st.selectbox,
        "multiselect": st.multiselect,
        "radio": st.radio,
    }

    if widget not in widget_mapper:
        raise ValueError(f"Unknown widget: {widget}")

    def format_func(x):
        return label if x is None else x

    options = [None, *options]

    return widget_mapper[widget](label=" ", options=options, format_func=format_func)
