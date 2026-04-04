"""
model_loader.py
---------------
Handles loading, preprocessing, inference and Grad-CAM
for both deployed models:

  Colon Dataset  →  EfficientNetB0  (Binary: ACA vs Normal)
  GI Dataset     →  ResNet50        (3-Class: Normal / Colitis / Polyps)

Both models are downloaded from Hugging Face on first request
and cached in memory for all subsequent requests.
"""

import os
import io
import base64
import tempfile
import numpy as np
import requests
import matplotlib.cm as cm_lib
import tensorflow as tf
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
IMG_SIZE = 224
HF_BASE  = "https://huggingface.co/kayiwarahim"

COLON_MODEL_URL = f"{HF_BASE}/D2_DenseNet121/resolve/main/DenseNet121_dataset2.keras"
GI_MODEL_URL    = f"{HF_BASE}/D1_DenseNet121/resolve/main/DenseNet121_dataset1.keras"

# Internal class folder names (from your Kaggle dataset)
COLON_LABELS  = ["colon_aca", "colon_n"]
GI_LABELS     = ["0_normal", "1_ulcerative_colitis", "2_polyps"]

# Display names shown to the user
COLON_DISPLAY = ["Normal","Adenocarcinoma"]
#COLON_DISPLAY = ["Adenocarcinoma", "Normal"]
GI_DISPLAY    = ["Normal", "Ulcerative Colitis", "Polyps"]

# ─────────────────────────────────────────────────────────────
# MODEL CACHE
# Both models download once and stay in memory
# ─────────────────────────────────────────────────────────────
_colon_model = None   # EfficientNetB0
_gi_model    = None   # ResNet50


def _download_model(url: str, label: str):
    """
    Downloads a .keras model from Hugging Face into a temp file,
    loads it with TensorFlow, deletes the temp file, returns the model.
    Uses streaming so large files don't cause memory issues.
    """
    print(f"\nDownloading {label} model from Hugging Face...")
    print(f"  URL: {url}")

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {label} model.\n"
            f"HTTP status: {response.status_code}\n"
            f"Check that your file is uploaded and public on Hugging Face."
        )

    # Write to temp file — TensorFlow needs a file path to load from
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    print(f"  Saved temp file: {tmp_path}")
    print(f"  Loading into TensorFlow...")

    model = tf.keras.models.load_model(tmp_path, compile=False)

    # Delete temp file immediately after loading into memory
    os.remove(tmp_path)

    print(f"  {label} loaded — {model.count_params():,} parameters")
    return model


def load_colon_model():
    """
    Returns EfficientNetB0 colon model.
    Downloads from Hugging Face on first call, cached after that.
    """
    global _colon_model
    if _colon_model is None:
        _colon_model = _download_model(
            COLON_MODEL_URL,
            "Colon — EfficientNetB0 (Binary)"
        )
    return _colon_model


def load_gi_model():
    """
    Returns ResNet50 GI model.
    Downloads from Hugging Face on first call, cached after that.
    """
    global _gi_model
    if _gi_model is None:
        _gi_model = _download_model(
            GI_MODEL_URL,
            "GI — ResNet50 (3-Class)"
        )
    return _gi_model


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# Each backbone was trained with its own specific preprocessing.
# Using the wrong one will give wrong or random predictions.
# ─────────────────────────────────────────────────────────────
def preprocess_colon(pil_img):
    """
    EfficientNetB0 preprocessing.
    Scales pixel values to [-1, 1] range.
    """
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr.copy())
    return np.expand_dims(arr, axis=0), np.array(img)


def preprocess_gi(pil_img):
    """
    ResNet50 preprocessing.
    Subtracts ImageNet channel means [103.939, 116.779, 123.68],
    converts RGB to BGR.
    """
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr.copy())
    return np.expand_dims(arr, axis=0), np.array(img)


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# Works for all transfer learning models by going inside
# the backbone (model.layers[1]) to find the last Conv2D.
# ─────────────────────────────────────────────────────────────
def make_gradcam(model, img_array, pred_index=None):
    """
    Computes Grad-CAM heatmap for any transfer learning model.
    Returns a 2D numpy array of values between 0 and 1,
    or None if computation fails.
    """
    try:
        backbone  = model.layers[1]
        last_conv = None

        # Find the last Conv2D layer inside the backbone
        for layer in reversed(backbone.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break

        if last_conv is None:
            print("Grad-CAM: no Conv2D found in backbone")
            return None

        # Build gradient model using backbone inputs/outputs
        grad_model = tf.keras.models.Model(
            inputs=backbone.inputs,
            outputs=[last_conv.output, backbone.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, backbone_out = grad_model(img_array)
            tape.watch(conv_outputs)

            # Pass backbone output through remaining head layers
            x = backbone_out
            for layer in model.layers[2:]:
                x = layer(x)
            predictions = x

            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))

            class_channel = predictions[:, pred_index]

        grads   = tape.gradient(class_channel, conv_outputs)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def overlay_gradcam(original_arr, heatmap, alpha=0.45):
    """
    Overlays Grad-CAM heatmap onto original image using jet colormap.
    Returns numpy array of same shape as original_arr.
    """
    if heatmap is None:
        return original_arr

    size        = (original_arr.shape[1], original_arr.shape[0])
    heatmap_u8  = np.uint8(255 * heatmap)
    jet_colors  = cm_lib.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = np.uint8(jet_colors[heatmap_u8] * 255)
    jet_img     = np.array(Image.fromarray(jet_heatmap).resize(size))
    return np.uint8(jet_img * alpha + original_arr * (1 - alpha))


def arr_to_base64(arr: np.ndarray) -> str:
    """Converts a numpy image array to base64 PNG string for API response."""
    pil_img = Image.fromarray(arr.astype(np.uint8))
    buffer  = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# Called by FastAPI endpoint for every image upload
# ─────────────────────────────────────────────────────────────
def predict_image(image_bytes: bytes, dataset_type: str) -> dict:
    """
    Main prediction function.

    Parameters
    ----------
    image_bytes  : raw bytes of the uploaded image file
    dataset_type : "colon" or "gi"

    Returns
    -------
    dict with keys:
        prediction    — top predicted class display name
        confidence    — confidence % of top prediction (float)
        all_probs     — dict of all class names → confidence %
        gradcam_image — base64 PNG string of Grad-CAM overlay
        original_image— base64 PNG string of original image
        model_used    — name of model used
        dataset_type  — echoes back the dataset_type input
    """

    # ── Open image ──────────────────────────────────────────
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file — could not open: {str(e)}"}

    # ── Select model and config ──────────────────────────────
    if dataset_type == "colon":
        model      = load_colon_model()
        preprocess = preprocess_colon
        display    = COLON_DISPLAY
        is_binary  = True
        model_name = "DenseNet121"

    elif dataset_type == "gi":
        model      = load_gi_model()
        preprocess = preprocess_gi
        display    = GI_DISPLAY
        is_binary  = False
        model_name = "DenseNet121"

    else:
        return {
            "error": (
                f"Unknown dataset_type '{dataset_type}'. "
                "Use 'colon' or 'gi'."
            )
        }

    # ── Preprocess ──────────────────────────────────────────
    img_array, original_arr = preprocess(pil_img)

    # ── Predict ─────────────────────────────────────────────
    preds = model.predict(img_array, verbose=0)

    if is_binary:
        #probs = preds[0]
        #pred_idx = int(np.argmax(probs))
        prob_pos = float(preds[0][0])
        probs    = [prob_pos, 1.0 - prob_pos]
        pred_idx = 0 if prob_pos > 0.5 else 1
    else:
        probs    = [float(p) for p in preds[0]]
        pred_idx = int(np.argmax(probs))

    pred_label = display[pred_idx]
    confidence = round(probs[pred_idx] * 100, 2)
    #added to predict non-colon images
    if confidence < 70:
        return {
        "prediction": "Unknown / Not a colon image",
        "confidence": confidence,
        "all_probs": all_probs,
        "gradcam_image": gradcam_b64,
        "original_image": original_b64,
        "model_used": model_name,
        "dataset_type": dataset_type,
    }
    all_probs  = {
        label: round(prob * 100, 2)
        for label, prob in zip(display, probs)
    }

    # ── Grad-CAM ─────────────────────────────────────────────
    heatmap        = make_gradcam(model, img_array, pred_index=pred_idx)
    gradcam_arr    = overlay_gradcam(original_arr, heatmap)
    gradcam_b64    = arr_to_base64(gradcam_arr)
    original_b64   = arr_to_base64(original_arr)

    # ── Return response ──────────────────────────────────────
    return {
        "prediction":     pred_label,
        "confidence":     confidence,
        "all_probs":      all_probs,
        "gradcam_image":  gradcam_b64,
        "original_image": original_b64,
        "model_used":     model_name,
        "dataset_type":   dataset_type,
    }
