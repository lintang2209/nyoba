import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import gdown
import json

# ====================
# CSS Styling Aman
# ====================
st.markdown("""
<style>
body { background-color: white; }

body::before {
    content: "";
    position: absolute;
    top: -30px; left: -50px;
    width: 250px; height: 250px;
    background: url('https://i.ibb.co/Lh2W1tV/leaf-top.png') no-repeat;
    background-size: contain;
    transform: rotate(20deg);
    z-index: -1;
}

body::after {
    content: "";
    position: absolute;
    bottom: -30px; right: -50px;
    width: 250px; height: 250px;
    background: url('https://i.ibb.co/Z2ShYDC/leaf-bottom.png') no-repeat;
    background-size: contain;
    transform: rotate(-15deg);
    z-index: -1;
}

.center { text-align: center; padding-top: 120px; }
.title { font-size: 36px; font-weight: 700; color: #4b8b64; }
.subtitle { font-size: 16px; font-style: italic; color: #7d7d7d; margin-top: -10px; }

.stButton>button {
    background-color: #f0f0f0; color: #4b8b64;
    border-radius: 20px; border: none; padding: 10px 25px;
    font-weight: 600; cursor: pointer; transition: 0.3s;
}
.stButton>button:hover { background-color: #4b8b64; color: white; }
</style>
""", unsafe_allow_html=True)

# ====================
# Navigasi sederhana
# ====================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown("""
    <div class="center">
        <p class="title">ayo cek tanamanmu!</p>
        <p class="subtitle">kenali soybean rust sejak dini<br>untuk hasil panen yang lebih baik</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("cek disini"):
            st.session_state.page = "deteksi"
            st.rerun()

# ====================
# Halaman Deteksi (CNN & YOLO)
# ====================
elif st.session_state.page == "deteksi":

    st.title("Perbandingan Deteksi Penyakit Soybean Rust (CNN vs YOLO) 🌱")
    st.write("Unggah satu gambar daun kedelai untuk melihat hasil deteksi dari kedua model secara bersamaan.")

    # ---- Load CNN & class_names ----
    @st.cache_resource
    def load_cnn_model():
        GOOGLE_DRIVE_FILE_ID = "1sZegfJRnGu2tr00qtinTAeZeLaQnllrO"
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/cnn.h5")
        CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "models/class_names.json")

        os.makedirs(os.path.dirname(MODEL
