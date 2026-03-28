import streamlit as st
from api.predict import predict_image
import io
from PIL import Image

st.title("Automated Multi-Class Gastrointestinal Anomaly Detection in Colonoscopy Imagery")
st.write("Upload a colonoscopy image to get a prediction.")

# File uploader
uploaded_file = st.file_uploader("Upload Colonoscopy Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    # Run prediction
    result = predict_image(buf.read())
    st.success(f"Disease Predicted: {result['prediction']}")
