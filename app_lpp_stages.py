import os
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError

# TensorFlow completo incluye tf.lite. En Streamlit Cloud suele ser lo más estable.
# Si más adelante usás tflite-runtime, podés cambiar este import.
import tensorflow as tf

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
IMG_SIZE = 224
MODEL_PATH = "modelo_lpp_mobilenet.tflite"
CLASS_NAMES = ["stage1", "stage2", "stage3", "stage4"]

STAGE_INFO = {
    "stage1": {
        "roman": "Estadio I",
        "name": "Eritema no blanqueable",
        "emoji": "🟡",
        "description": "Piel intacta con enrojecimiento no blanqueable. La zona puede presentar dolor, firmeza, suavidad o diferencia de temperatura respecto al tejido adyacente.",
        "alert": "✅ Iniciar medidas preventivas: cambios posturales cada 2 h, superficie redistributiva de presión, hidratación de la piel.",
    },
    "stage2": {
        "roman": "Estadio II",
        "name": "Pérdida parcial del espesor",
        "emoji": "🟠",
        "description": "Pérdida parcial del espesor de la piel con exposición de la dermis. Lecho viable, rosado o rojo, húmedo. Puede presentarse como vesícula intacta o rota.",
        "alert": "⚠️ Evaluar con equipo clínico. Iniciar protocolo de curas y proteger bordes de la herida.",
    },
    "stage3": {
        "roman": "Estadio III",
        "name": "Pérdida total del espesor",
        "emoji": "🔴",
        "description": "Pérdida total del espesor de la piel con grasa subcutánea visible. Sin exposición de hueso, tendón ni músculo. Puede haber esfacelo y/o tejido necrótico.",
        "alert": "🔴 Derivar a equipo de heridas. Requiere desbridamiento y tratamiento especializado.",
    },
    "stage4": {
        "roman": "Estadio IV",
        "name": "Pérdida total de tejidos",
        "emoji": "🚨",
        "description": "Exposición o palpación directa de hueso, tendón o músculo. Frecuentemente incluye tunelizaciones, socavaciones y tejido necrótico extenso.",
        "alert": "🚨 CRÍTICO — Intervención médica urgente. Riesgo de complicaciones sistémicas (osteomielitis, sepsis). Notificar al médico responsable.",
    },
}

DISPLAY_NAMES = {
    "stage1": "Estadio I",
    "stage2": "Estadio II",
    "stage3": "Estadio III",
    "stage4": "Estadio IV",
}

# ── Optional Google Drive download ────────────────────────────────────────────
def maybe_download_model_from_drive():
    """
    Opcional: si publicás en Streamlit Cloud y no querés subir el .tflite al repo,
    podés definir en .streamlit/secrets.toml:

    GDRIVE_FILE_ID = "tu_file_id"

    y agregar gdown al requirements.txt.
    """
    if os.path.exists(MODEL_PATH):
        return None

    file_id = st.secrets.get("GDRIVE_FILE_ID", None) if hasattr(st, "secrets") else None
    if not file_id:
        return f"No se encontró `{MODEL_PATH}` en la carpeta de la app."

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Descargando modelo TFLite desde Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=True)
        if not os.path.exists(MODEL_PATH):
            return "No se pudo descargar el modelo desde Google Drive."
        return None
    except Exception as e:
        return f"Error descargando modelo desde Google Drive: {e}"

# ── TFLite model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_tflite_model():
    err = maybe_download_model_from_drive()
    if err:
        return None, None, None, err

    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details, None
    except Exception as e:
        return None, None, None, str(e)


def preprocess(img: Image.Image, input_details) -> np.ndarray:
    """Prepara la imagen respetando el tipo de entrada del modelo TFLite."""
    input_info = input_details[0]
    input_dtype = input_info["dtype"]

    arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)

    # IMPORTANTE: mantener igual que el entrenamiento original.
    # Tu app anterior usaba /255.0, por eso se mantiene.
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Si el modelo fue cuantizado a uint8/int8, convertir usando scale/zero_point.
    if input_dtype in (np.uint8, np.int8):
        scale, zero_point = input_info.get("quantization", (0.0, 0))
        if scale and scale > 0:
            arr = arr / scale + zero_point
        arr = np.clip(arr, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
        arr = arr.astype(input_dtype)
    else:
        arr = arr.astype(input_dtype)

    return arr


def predict_tflite(interpreter, input_details, output_details, image: Image.Image) -> np.ndarray:
    input_tensor = preprocess(image, input_details)

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Si la salida está cuantizada, des-cuantizar.
    output_info = output_details[0]
    if output_info["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_info.get("quantization", (0.0, 0))
        if scale and scale > 0:
            output = (output.astype(np.float32) - zero_point) * scale

    output = output.astype(np.float32)

    # Si por algún motivo la salida no viene normalizada, aplicar softmax.
    if not np.isclose(np.sum(output), 1.0, atol=1e-2):
        exp = np.exp(output - np.max(output))
        output = exp / np.sum(exp)

    return output

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Clasificador LPP")
st.caption("Lesiones por Presión · MobileNetV2 TFLite · Estadios I–IV")
st.divider()

# ── Model load ────────────────────────────────────────────────────────────────
interpreter, input_details, output_details, err = load_tflite_model()
if err:
    st.error(
        f"Modelo TFLite no encontrado o no cargable. Colocá `{MODEL_PATH}` en la misma carpeta "
        f"o configurá `GDRIVE_FILE_ID` en Streamlit Secrets.\n\n`{err}`"
    )
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
try:
    image = Image.open(uploaded)
except UnidentifiedImageError:
    st.error("No se pudo leer la imagen. Probá con un archivo JPG, PNG o WEBP válido.")
    st.stop()

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(image, use_container_width=True)

# ── Predict ───────────────────────────────────────────────────────────────────
with st.spinner("Analizando imagen..."):
    preds = predict_tflite(interpreter, input_details, output_details, image)

if len(preds) != len(CLASS_NAMES):
    st.error(
        f"La salida del modelo tiene {len(preds)} clases, pero la app espera {len(CLASS_NAMES)}. "
        "Revisá CLASS_NAMES para que coincida con el entrenamiento."
    )
    st.stop()

idx = int(np.argmax(preds))
pred_class = CLASS_NAMES[idx]
confidence = float(preds[idx])
info = STAGE_INFO[pred_class]

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
    p = float(prob)
    lbl = f"{'▶ ' if cls == pred_class else '   '}{DISPLAY_NAMES.get(cls, cls)}"
    st.progress(p, text=f"{lbl} — {p*100:.1f}%")

st.divider()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.caption(
    "⚕️ Herramienta de apoyo clínico asistido por IA. "
    "No reemplaza el juicio de un profesional de la salud. "
    "Toda clasificación debe ser validada por personal médico o de enfermería calificado."
)
