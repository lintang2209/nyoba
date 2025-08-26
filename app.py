import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from ultralytics import YOLO
import json
from PIL import Image

# ====================
# Load CNN Model dari Google Drive
# ====================
@st.cache_resource
def load_cnn_model():
    import gdown
    import tensorflow as tf
    import os

        # --- Unduh model dari Google Drive ---
        GOOGLE_DRIVE_FILE_ID = "11JeSvrid8Zw2xurG-pciDrw6EdI2qXuAd" # Link sudah disesuaikan
        MODEL_PATH = "models/cnn.h5"
        
        # Periksa apakah folder "models" ada, jika tidak, buatlah
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            st.info("Mengunduh model dari Google Drive...")
            try:
                gdown.download(f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}', MODEL_PATH, quiet=False)
                st.success("Model berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal mengunduh model dari Google Drive: {e}")
                return None

        # --- Muat model ---
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model CNN: {e}")
            return None

# ===============================
# Load YOLO model & class names
# ===============================
@st.cache_resource
def load_yolo_model():
    YOLO_PATH = "models/best.pt"  # tetap lokal
    if not os.path.exists(YOLO_PATH):
        st.error(f"Model YOLO tidak ditemukan: {YOLO_PATH}")
        return None
    return YOLO(YOLO_PATH)

@st.cache_resource
def load_class_names():
    CLASS_PATH = "models/class_names.json"  # tetap lokal
    if not os.path.exists(CLASS_PATH):
        st.error(f"class_names.json tidak ditemukan: {CLASS_PATH}")
        return None
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

# ===============================
# Preprocess untuk CNN
# ===============================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===============================
# Prediksi CNN
# ===============================
def predict_cnn(model, img, class_names):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    return class_names[class_idx], confidence

# ===============================
# Prediksi YOLO
# ===============================
def predict_yolo(model, img):
    results = model.predict(img)
    return results

# ===============================
# Streamlit UI
# ===============================
st.title("🌿 Deteksi Daun: CNN vs YOLO")

cnn_model = load_cnn_model()
yolo_model = load_yolo_model()
class_names = load_class_names()

uploaded_file = st.file_uploader("Upload gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and cnn_model and yolo_model and class_names:
    image = Image.open(uploaded_file).convert("RGB")

    # CNN Prediction
    label_cnn, confidence_cnn = predict_cnn(cnn_model, image, class_names)
    
    # YOLO Prediction
    results = predict_yolo(yolo_model, uploaded_file)
    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Tampilkan hasil berdampingan
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 CNN Prediction")
        st.image(image, caption="Gambar Input", use_container_width=True)
        st.markdown(f"**Hasil Prediksi CNN:** {label_cnn}")
        st.markdown(f"**Confidence:** {confidence_cnn:.2f}")

    with col2:
        st.subheader("🔹 YOLO Prediction")
        st.image(result_img, caption="Hasil YOLO", use_container_width=True)
