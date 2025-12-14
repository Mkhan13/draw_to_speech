import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from .backend import DoodleModel

def run_frontend():
    st.title("Draw to Speech AAC")
    st.write("Draw a doodle below, then click **Identify**.")

    data = np.load("./data/processed/train.npz")
    class_names = data["class_names"] # Load class names

    @st.cache_resource
    def load_model():
        return DoodleModel("best_effnet.pth", class_names) # Load model once

    model = load_model()

    stroke_color = st.color_picker("Choose drawing color", "#000000")
    stroke_width = st.slider("Brush size", 3, 20, 8)

    canvas_result = st_canvas( # Drawing canvas
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="white",
        width=350,
        height=350,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Identify"):
        if canvas_result.image_data is None:
            st.warning("Please draw something first.")
            return

        img = canvas_result.image_data[:, :, :3]
        img_gray = np.mean(img, axis=2)

        pred = model.predict(img_gray)
        st.subheader(f"Prediction: **{pred}**")