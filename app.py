
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import gdown
import json

# ====================
# CSS Styling dengan URL daun online
# ====================
st.markdown("""
    <style>
        .main { background-color: white; }

        body::before {
            content: "";
            position: absolute;
            top: -30px; left: -50px;
            width: 250px; height: 250px;
            background: url("https://i.ibb.co/Lh2W1tV/leaf-top.png") no-repeat;
            background-size: contain;
            transform: rotate(20deg);
            z-index: -1;
        }

        body::after {
            content: "";
            position: absolute;
            bottom: -30px; right: -50px;
            width: 250px; height: 250px;
            background: url("https://i.ibb.co/Z2ShYDC/leaf-bottom.png") no-repeat;
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
        MODEL_PATH = "models/cnn.h5"
        CLASS_NAMES_PATH = "models/class_names.json"

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            st.info("Mengunduh model dari Google Drive...")
            gdown.download(f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}', MODEL_PATH, quiet=False)
            st.success("Model berhasil diunduh!")

        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model CNN: {e}")
            return None, None

        # Load class names
        try:
            with open(CLASS_NAMES_PATH, "r") as f:
                class_names = json.load(f)
        except Exception as e:
            st.error(f"Gagal memuat class names: {e}")
            return model, None

        return model, class_names

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

    cnn_model, class_names = load_cnn_model()
    yolo_model = load_yolo_model()

    if cnn_model is None or class_names is None or yolo_model is None:
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
                img_resized = image.resize((224,224))
                img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

                prediction = cnn_model.predict(img_array)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                predicted_class_name = class_names[class_id]

                st.write(f"### Prediksi: **{predicted_class_name}**")
                st.write(f"Confidence: **{confidence:.2f}**")
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
        st.rerun()
