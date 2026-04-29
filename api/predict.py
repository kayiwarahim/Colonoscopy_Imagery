import numpy as np
import requests
import matplotlib.cm as cm_lib
import tensorflow as tf
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
IMG_SIZE = 224
NUM_CLASSES = 3

MODEL_URL = "https://huggingface.co/kayiwarahim/mobilenet_v2/resolve/main/mobilenet_v2.h5"

# =====================================================
# LOAD MODEL DIRECTLY FROM HUGGING FACE
# =====================================================
_model = None  # cached model

def load_model_from_hf():
    global _model
    if _model is None:
        print("Loading Keras model from Hugging Face...")
        response = requests.get(MODEL_URL)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch model: {response.status_code}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

    print(f"  Saved temp file: {tmp_path}")
    print(f"  Loading into TensorFlow...")

    model = tf.keras.models.load_model(tmp_path, compile=False)

    # Delete temp file immediately after loading into memory
    os.remove(tmp_path)

    return _model

# =====================================================
# CLASS LABELS
# =====================================================
class_names = ["0_normal", "1_ulcerative_colitis", "2_polyps"]

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_image(image_bytes):
    model = load_model_from_hf()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return {'error': 'Invalid image file'}

    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds, axis=1)[0]]

    return {'prediction': pred_class}
