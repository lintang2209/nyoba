import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import json
import gdown  # untuk download model dari Google Drive

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
.maybe {color:#b26a00; font-weight:700;}
</style>
""", unsafe_allow_html=True)

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

elif st.session_state.page == "deteksi":
    st.title("Deteksi Penyakit Soybean Rust (CNN + YOLO) 🌱")
    st.write("Unggah gambar daun untuk melihat hasil deteksi CNN + YOLO")

    @st.cache_resource
    def download_cnn_model():
        url = "https://drive.google.com/uc?id=1JeSvrid8Zw2xurG-pciDrw6EdI2qXuAd"
        local_path = "models/cnn.h5"
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(local_path):
            with st.spinner("Mengunduh model CNN dari Google Drive..."):
                gdown.download(url, local_path, quiet=False)
        return local_path

    @st.cache_resource
    def load_cnn_model():
        MODEL_PATH = download_cnn_model()
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model CNN tidak ditemukan: {MODEL_PATH}")
            return None
        with st.spinner("Memuat model CNN..."):
            return tf.keras.models.load_model(MODEL_PATH)

    @st.cache_resource
    def load_yolo_model():
        MODEL_PATH = "models/best.pt"
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model YOLOv8 tidak ditemukan: {MODEL_PATH}")
            return None
        with st.spinner("Memuat model YOLOv8..."):
            return YOLO(MODEL_PATH)

    @st.cache_resource
    def load_class_names():
        PATH = "models/class_names.json"
        if not os.path.exists(PATH):
            st.error(f"File class_names.json tidak ditemukan: {PATH}")
            return None
        with open(PATH, "r") as f:
            return json.load(f)

    def model_has_rescaling(m):
        try:
            for lyr in m._flatten_layers():
                if isinstance(lyr, tf.keras.layers.Rescaling):
                    return True
        except Exception:
            for lyr in m.layers:
                if isinstance(lyr, tf.keras.layers.Rescaling):
                    return True
        return False

    cnn_model = load_cnn_model()
    yolo_model = load_yolo_model()
    class_names = load_class_names()
    if None in [cnn_model, yolo_model, class_names]:
        st.stop()

    HAS_RESCALING = model_has_rescaling(cnn_model)
    st.caption(f": {'(Rescaling di dalam model → JANGAN /255 di app)' if HAS_RESCALING else '(Tidak ada Rescaling di model → /255 di app)'}")

    uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg","png","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.write("---")

        img_resized = image.resize((224,224))
        x = np.array(img_resized, dtype=np.float32)
        if HAS_RESCALING:
            img_array = np.expand_dims(x, axis=0)
        else:
            img_array = np.expand_dims(x / 255.0, axis=0)

        with st.spinner("Menjalankan prediksi CNN..."):
            prediction = cnn_model.predict(img_array)
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        THRESHOLD = 0.60
        cnn_label = class_names[class_id] if confidence >= THRESHOLD else "Tidak yakin"

        with st.spinner("Menjalankan deteksi YOLO..."):
            results = yolo_model(image)
            results_img = results[0].plot()
            yolo_detected = len(results[0].boxes) > 0

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hasil CNN")
            st.write("Raw prediction:", prediction)
            st.write("Class names:", class_names)
            st.write(f"Predicted class id: {class_id}")
            st.write(f"Confidence: {confidence:.2f}")
            st.write(f"Label CNN: **{cnn_label}**")

        with col2:
            st.subheader("Hasil YOLOv8")
            st.image(results_img, caption="Hasil YOLO", use_column_width=True)
            if yolo_detected:
                st.success("Lesion terdeteksi oleh YOLO")
            else:
                st.info("Tidak ada lesion terdeteksi oleh YOLO")

        if not yolo_detected:
            if cnn_label.lower() in ["soybean rust", "sakit"] and confidence >= 0.99:
                final_label = ("Sakit", "bad")
            else:
                final_label = ("Sehat", "good")
        else:
            final_label = ("Sakit", "bad")

        st.markdown("---")
        st.subheader("🌱 Prediksi Akhir")
        st.markdown(f"<span class='{final_label[1]}'>{final_label[0]}</span>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()
