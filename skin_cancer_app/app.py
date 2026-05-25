import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Configuración ────────────────────────────────────────────────────────────
IMG_SIZE = 224
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL = {
    'akiec': 'Queratosis Actínica',
    'bcc':   'Carcinoma Basocelular',
    'bkl':   'Lesión Queratósica Benigna',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Nevus Melanocítico',
    'vasc':  'Lesiones Vasculares'
}
COLORS = {
    'mel':   '🔴',
    'bcc':   '🟠',
    'akiec': '🟡',
    'bkl':   '🟡',
    'nv':    '🟢',
    'df':    '🟢',
    'vasc':  '🟢',
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mejor_modelo.h5')

def preprocess_image(image):
    img = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Clasificador de Lesiones de Piel", page_icon="🔬", layout="centered")

st.title("🔬 Clasificador de Lesiones de Piel")
st.markdown("Sube una imagen de una lesión cutánea para obtener un diagnóstico preliminar basado en deep learning.")
st.warning("⚠️ Esta herramienta es solo de apoyo diagnóstico. No reemplaza la opinión de un médico.")

model = load_model()

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagen cargada", use_column_width=True)
    
    with col2:
        with st.spinner("Analizando..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)[0]
            top_idx = np.argmax(predictions)
            top_class = CLASS_NAMES[top_idx]
            top_prob = predictions[top_idx]

        st.subheader("Resultado")
        st.markdown(f"### {COLORS[top_class]} {CLASS_FULL[top_class]}")
        st.markdown(f"**Confianza: {top_prob*100:.1f}%**")
        
        st.subheader("Probabilidades por clase")
        sorted_idx = np.argsort(predictions)[::-1]
        for idx in sorted_idx:
            cls = CLASS_NAMES[idx]
            prob = predictions[idx]
            st.markdown(f"{COLORS[cls]} **{cls}** — {CLASS_FULL[cls]}")
            st.progress(float(prob), text=f"{prob*100:.1f}%")