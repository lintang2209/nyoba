import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import json
import gc

# ====================
# Styling CSS
# ====================
st.markdown("""
<style>
body {background-color: white;}
.center {text-align: center; padding-top: 120px;}
.title {font-size: 36px; font-weight: 700; color: #4b8b64;}
.subtitle {font-size: 16px; font-style: italic; color: #7d7d7d; margin-top: -10px;}
.stButton>button {
    background-color: #f0f0f0; color: #4b8b64; border-radius: 20px; border: none;
    padding: 10px 25px; font-weight: 600; cursor: pointer; transition: 0.3s;
}
.stButton>button:hover {background-color: #4b8b64; color: white;}
.good {color:#1b7f2a; font-weight:700;}
.bad {color:#c62828; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ====================
# Navigasi
# ====================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown("""
    <div class="center">
        <p class="title">Ayo cek tanamanmu!</p>
        <p class="subtitle">Kenali soybean rust sejak dini untuk hasil panen yang lebih baik</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Cek Disini"):
            st.session_state.page = "deteksi"
            st.rerun()

elif st.session_state.page == "deteksi":
    st.title("Deteksi Penyakit Soybean Rust (CNN + YOLO) 🌱")
    st.write("Unggah gambar daun untuk melihat hasil deteksi CNN + YOLO")

    # ====================
    # Load Models
    # ====================
    @st.cache_resource
    def load_cnn_model():
        MODEL_PATH = "models/cnn.h5"
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model CNN tidak ditemukan: {MODEL_PATH}")
            return None
        return tf.keras.models.load_model(MODEL_PATH)

    @st.cache_resource
    def load_yolo_model():
        MODEL_PATH = "models/best.pt"  # sesuaikan nama file YOLO
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model YOLOv8 tidak ditemukan: {MODEL_PATH}")
            return None
        return YOLO(MODEL_PATH)

    @st.cache_resource
    def load_class_names():
        PATH = "models/class_names.json"
        if not os.path.exists(PATH):
            st.error(f"File class_names.json tidak ditemukan: {PATH}")
            return None
        with open(PATH, "r") as f:
            return json.load(f)

    cnn_model = load_cnn_model()
    yolo_model = load_yolo_model()
    class_names = load_class_names()
    if None in [cnn_model, yolo_model, class_names]:
        st.stop()

    # ====================
    # Upload Gambar
    # ====================
    uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg","png","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((1024,1024))  # batasi ukuran agar hemat memori
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.write("---")

        # ====================
        # CNN Prediction
        # ====================
        img_resized = image.resize((224,224))
        x = np.array(img_resized, dtype=np.float32)
        # ❌ Jangan /255, karena model sudah ada Rescaling
        img_array = np.expand_dims(x, axis=0)

        prediction = cnn_model.predict(img_array, verbose=0)[0]
        class_id = int(np.argmax(prediction))
        confidence = float(prediction[class_id])
        cnn_label = class_names[class_id]

        # ====================
        # YOLO Prediction
        # ====================
        results = yolo_model(image)
        results_img = results[0].plot()
        yolo_detected = len(results[0].boxes) > 0

        # ====================
        # Layout
        # ====================
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hasil CNN")
            st.write(f"Prediksi: **{cnn_label}**")
            st.write(f"Confidence: **{confidence:.2f}**")

        with col2:
            st.subheader("Hasil YOLO")
            st.image(results_img, caption="Hasil YOLO", use_column_width=True)
            if yolo_detected:
                st.success("Lesion terdeteksi oleh YOLO")
            else:
                st.info("Tidak ada lesion terdeteksi oleh YOLO")

        # ====================
        # Bersihkan memori
        # ====================
        del results_img, results, img_array, x
        gc.collect()

        # ====================
        # Prediksi Gabungan CNN+YOLO
        # ====================
        final_label = ("Sakit", "bad") if yolo_detected or cnn_label.lower() == "soybean_rust" else ("Sehat", "good")
        st.markdown("---")
        st.subheader("🌱 Prediksi Akhir")
        st.markdown(f"<span class='{final_label[1]}'>{final_label[0]}</span>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()
