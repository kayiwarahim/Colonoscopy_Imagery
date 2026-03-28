import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os, io, requests, tempfile
from PIL import Image
import io, requests

# =====================================================
# CONFIGURATION
# =====================================================
IMG_SIZE = 224
NUM_CLASSES = 3

# Hugging Face URL of your model file (.h5 format)
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

        # Write to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load model from the temp file
        _model = load_model(tmp_path)
        print("Model loaded successfully!")

        # Optional: clean up temp file
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

    # Preprocess image
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # normalize

    # Predict
    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds, axis=1)[0]]

    return {'prediction': pred_class}
