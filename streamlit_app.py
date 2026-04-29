import io
import base64
from unittest import result
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from api.predict import predict_image

def base64_to_image(base64_str):
    """Convert base64 string back to PIL image"""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))

def main():
    st.title("Automated Multi-Class Gastrointestinal Anomaly Detection")
    st.write("Upload a colonoscopy image to get predictions.")

    # ✅ DATASET SELECTOR (NEW)
    dataset_type = st.selectbox(
        "Select Model",
        ["colon", "gi"]
    )

    uploaded_file = st.file_uploader(
        "Upload Colonoscopy Image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        return

    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width='stretch')

    # Convert to bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    
    # ✅ Show loading instead of white screen
    with st.spinner("🔄 Loading model and analyzing image..."):
        result = predict_image(buf.read(), dataset_type)

    # Handle errors
    if "error" in result:
        st.error(result["error"])
        return

    # ✅ DISPLAY RESULTS
    if "Unknown" in result["prediction"]:
        st.warning("⚠️ This does not appear to be a valid colon image.")
    else:
        st.success(f"Prediction: {result['prediction']}")
    #st.success(f"Prediction: {result['prediction']}")
    st.info(f"Confidence: {result['confidence']}%")
    st.write(f"Model Used: {result['model_used']}")

    # ✅ BAR CHART (NEW 🔥)
    st.subheader("Class Probabilities")
    probs = result["all_probs"]

    labels = list(probs.keys())
    values = list(probs.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim([0, 100])
    st.pyplot(fig)

    # ✅ SHOW GRAD-CAM (NEW 🔥)
    st.subheader("Grad-CAM Visualization")

    gradcam_img = base64_to_image(result["gradcam_image"])
    original_img = base64_to_image(result["original_image"])

    col1, col2 = st.columns(2)

    with col1:
        st.image(original_img, caption="Original Image")

    with col2:
        st.image(gradcam_img, caption="Grad-CAM Heatmap")

if __name__ == "__main__":
    main()