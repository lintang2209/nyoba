import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import gdown
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Class names sesuai dataset
CLASS_NAMES = ["sehat", "Soybean Rust"]

# ====================
# CSS Styling tetap sama
# ====================
st.markdown("""
    <style>
        /* ... CSS kamu tetap di sini ... */
    </style>
""", unsafe_allow_html=True)

# ====================
# Navigasi sederhana tetap sama
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

    if st.button("cek disini"):
        st.session_state.page = "deteksi"
        st.experimental_rerun()

elif st.session_state.page == "deteksi":

    st.title("Perbandingan Deteksi Penyakit Soybean Rust (CNN vs YOLO) 🌱")
    st.write("Unggah satu gambar daun kedelai untuk melihat hasil deteksi dari kedua model secara bersamaan.")

    @st.cache_resource
    def load_cnn_model():
        GOOGLE_DRIVE_FILE_ID = "1sZegfJRnGu2tr00qtinTAeZeLaQnllrO"
        MODEL_PATH = "models/cnn.h5"
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            st.info("Mengunduh model dari Google Drive...")
            try:
                gdown.download(f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}', MODEL_PATH, quiet=False)
                st.success("Model berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal mengunduh model dari Google Drive: {e}")
                return None
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model CNN: {e}")
            return None

    @st.cache_resource
    def load_yolo_model():
        MODEL_PATH = "models/best.pt"
        if not os.path.exists(MODEL_PATH):
            st.error(f"File model YOLOv8 tidak ditemukan: {MODEL_PATH}")
            return None
        try:
            model = YOLO(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model YOLOv8: {e}")
            return None

    cnn_model = load_cnn_model()
    yolo_model = load_yolo_model()

    if cnn_model is None or yolo_model is None:
        st.stop()

    uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.write("---")

        col1, col2 = st.columns(2)

        # ==== CNN ====
        with col1:
            st.header("Hasil Analisis CNN")
            try:
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized)
                img_array = preprocess_input(img_array)  # preprocessing MobileNetV2
                img_array = np.expand_dims(img_array, axis=0)

                prediction = cnn_model.predict(img_array)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)

                predicted_class_name = CLASS_NAMES[class_id]

                st.write(f"### Prediksi: **{predicted_class_name}**")
                st.write(f"Confidence: **{confidence:.2f}**")

                # Threshold confidence
                threshold = 0.6
                if confidence < threshold:
                    st.warning("Model kurang yakin dengan prediksi ini.")

            except Exception as e:
                st.error(f"Terjadi kesalahan pada model CNN: {e}")

        # ==== YOLOv8 ====
        with col2:
            st.header("Hasil Analisis YOLOv8")
            try:
                results = yolo_model(image)
                results_img = results[0].plot()
                st.image(results_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

                if len(results[0].boxes) > 0:
                    st.write("#### Detail Deteksi:")
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        st.write(f"- Ditemukan **Penyakit Soybean Rust** dengan confidence **{conf:.2f}**")
                else:
                    st.write("Tidak ditemukan penyakit Soybean Rust.")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada model YOLOv8: {e}")

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.experimental_rerun()
