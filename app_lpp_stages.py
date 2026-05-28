import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador LPP",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS mínimo seguro ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 660px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_NAMES = ["stage1", "stage2", "stage3", "stage4"]

STAGE_INFO = {
    "stage1": {
        "roman":       "Estadio I",
        "name":        "Eritema no blanqueable",
        "emoji":       "🟡",
        "description": "Piel intacta con enrojecimiento no blanqueable. La zona puede presentar dolor, firmeza, suavidad o diferencia de temperatura respecto al tejido adyacente.",
        "alert":       "✅ Iniciar medidas preventivas: cambios posturales cada 2 h, superficie redistributiva de presión, hidratación de la piel.",
    },
    "stage2": {
        "roman":       "Estadio II",
        "name":        "Pérdida parcial del espesor",
        "emoji":       "🟠",
        "description": "Pérdida parcial del espesor de la piel con exposición de la dermis. Lecho viable, rosado o rojo, húmedo. Puede presentarse como vesícula intacta o rota.",
        "alert":       "⚠️ Evaluar con equipo clínico. Iniciar protocolo de curas y proteger bordes de la herida.",
    },
    "stage3": {
        "roman":       "Estadio III",
        "name":        "Pérdida total del espesor",
        "emoji":       "🔴",
        "description": "Pérdida total del espesor de la piel con grasa subcutánea visible. Sin exposición de hueso, tendón ni músculo. Puede haber esfacelo y/o tejido necrótico.",
        "alert":       "🔴 Derivar a equipo de heridas. Requiere desbridamiento y tratamiento especializado.",
    },
    "stage4": {
        "roman":       "Estadio IV",
        "name":        "Pérdida total de tejidos",
        "emoji":       "🚨",
        "description": "Exposición o palpación directa de hueso, tendón o músculo. Frecuentemente incluye tunelizaciones, socavaciones y tejido necrótico extenso.",
        "alert":       "🚨 CRÍTICO — Intervención médica urgente. Riesgo de complicaciones sistémicas (osteomielitis, sepsis). Notificar al médico responsable.",
    },
}

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        m = tf.keras.models.load_model("modelo_lpp_mobilenet.keras")
        return m, None
    except Exception as e:
        return None, str(e)

def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Clasificador LPP")
st.caption("Lesiones por Presión · MobileNetV2 · Estadios I–IV")
st.divider()

# ── Model load ────────────────────────────────────────────────────────────────
model, err = load_model()
if err:
    st.error(f"Modelo no encontrado. Colocá `modelo_lpp_mobilenet.keras` en la misma carpeta.\n\n`{err}`")
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Subir imagen de la lesión",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.info("Subí una foto de la lesión para obtener la clasificación del estadio.")
    st.stop()

# ── Image display ─────────────────────────────────────────────────────────────
image = Image.open(uploaded)
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(image, use_container_width=True)

# ── Predict ───────────────────────────────────────────────────────────────────
with st.spinner("Analizando imagen..."):
    preds      = model.predict(preprocess(image), verbose=0)[0]

idx        = int(np.argmax(preds))
pred_class = CLASS_NAMES[idx]
confidence = float(preds[idx])
info       = STAGE_INFO[pred_class]

st.divider()

# ── Result ────────────────────────────────────────────────────────────────────
col_stage, col_conf = st.columns([3, 1])
with col_stage:
    st.subheader(f"{info['emoji']} {info['roman']} — {info['name']}")
with col_conf:
    st.metric("Confianza", f"{confidence*100:.1f}%")

st.caption(info["description"])

# Baja confianza
if confidence < 0.60:
    st.warning(f"⚡ Confianza baja ({confidence*100:.1f}%). Resultado orientativo — verificar con criterio clínico.")

# Alerta clínica
if pred_class in ("stage3", "stage4"):
    st.error(info["alert"])
elif pred_class == "stage2":
    st.warning(info["alert"])
else:
    st.success(info["alert"])

st.divider()

# ── Probability bars ──────────────────────────────────────────────────────────
st.caption("**Distribución de probabilidad por estadio**")

for cls, prob in zip(CLASS_NAMES, preds):
    p   = float(prob)
    lbl = f"{'▶ ' if cls == pred_class else '   '}{cls}"
    st.progress(p, text=f"{lbl}  —  {p*100:.1f}%")

st.divider()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.caption(
    "⚕️ Herramienta de apoyo clínico asistido por IA. "
    "No reemplaza el juicio de un profesional de la salud. "
    "Toda clasificación debe ser validada por personal médico o de enfermería calificado."
)
