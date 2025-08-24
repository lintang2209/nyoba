import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import json

# ====================
# CSS Styling (sama seperti sebelumnya)
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
# Halaman Deteksi
# ====================
elif st.session_state.page == "deteksi":
    st.title("Deteksi Penyakit Soybean Rust (CNN + YOLO) 🌱")
    st.write("Unggah gambar daun untuk melihat hasil deteksi CNN + YOLO")

    @st.cache_resource
    def load_cnn_model():
        MODEL_PATH = "models/cnn.h5"
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model CNN tidak ditemukan: {MODEL_PATH}")
            return None
        return tf.keras.models.load_model(MODEL_PATH)

    @st.cache_resource
    def load_yolo_model():
        MODEL_PATH = "models/best.pt"
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

    uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg","png","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.write("---")

        # ==== Preprocess CNN ====
        img_resized = image.resize((224,224))
        img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)
        prediction = cnn_model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        # Threshold confidence
        if confidence < 0.6:
            cnn_label = "Tidak yakin"
        else:
            cnn_label = class_names[class_id]

        # ==== Jalankan YOLO ====
        results = yolo_model(image)
        results_img = results[0].plot()
        yolo_detected = len(results[0].boxes) > 0

        # ==== Layout 2 kolom ====
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hasil CNN")
            st.write("Raw prediction:", prediction)
            st.write("Predicted class id:", class_id)
            st.write("Class names:", class_names)
            st.write(f"Confidence: {confidence:.2f}")
            st.write(f"Label CNN: **{cnn_label}**")

        with col2:
            st.subheader("Hasil YOLOv8")
            st.image(results_img, caption="Hasil YOLO", use_column_width=True)
            if yolo_detected:
                st.success("Lesion terdeteksi oleh YOLO")
            else:
                st.info("Tidak ada lesion terdeteksi oleh YOLO")

        # ==== Prediksi akhir ====
        if cnn_label == "sehat" and not yolo_detected:
            final_label = "Sehat"
        elif cnn_label == "Soybean Rust" or yolo_detected:
            final_label = "Sakit"
        else:
            final_label = "Perlu dicek"

        st.markdown("---")
        st.subheader("🌱 Prediksi Akhir")
        st.write(f"**{final_label}**")
        if final_label == "Perlu dicek":
            st.warning("Confidence CNN rendah dan YOLO tidak mendeteksi lesion, harap periksa manual!")

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()
