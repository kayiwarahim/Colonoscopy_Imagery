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
    st.set_page_config(page_title="GastroAI", page_icon="images\\logo2.png", layout="wide")
    
    # custom CSS
    st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stButton > button { background-color: #2C7DA0; color: white; border-radius: 8px; }
    div.stFileUploader { border: 2px dashed #2C7DA0; border-radius: 12px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("GastroAI")
    st.image("images\\logo.png", width=150)
    st.markdown("### Automated Multi-Class Gastrointestinal Anomaly Detection")
    
    with st.sidebar:
        st.header("Settings")
        dataset_type = st.selectbox("Select Model", ["colon", "gi"])
        st.markdown("---")
        st.caption("Powered by DenseNet121 | Grad-CAM Explainability")
    
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Preview and quick results side by side
        col_left, col_right = st.columns(2)
        with col_left:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col_right:
            with st.spinner("Analyzing..."):
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                result = predict_image(buf.getvalue(), dataset_type)
            
            if "error" in result:
                st.error(result["error"])
                return
            
            st.success(f"**Prediction:** {result['prediction']}")
            st.info(f"**Confidence:** {result['confidence']}%")
            st.caption(f"Model: {result['model_used']}")
        
        # Expandable detailed analysis
        with st.expander("Detailed Analysis", expanded=True):
            # Bar chart using Plotly (or matplotlib as fallback)
            probs = result["all_probs"]
            import plotly.express as px
            fig = px.bar(x=list(probs.keys()), y=list(probs.values()),
                         title="Class Probabilities", labels={"x": "Class", "y": "Confidence (%)"},
                         color=list(probs.keys()), color_discrete_sequence=px.colors.sequential.Tealgrn)
            st.plotly_chart(fig, use_container_width=True)
            
            # Grad-CAM side by side
            st.subheader("Explainability: Grad-CAM Heatmap")
            grad_img = base64_to_image(result["gradcam_image"])
            orig_img = base64_to_image(result["original_image"])
            c1, c2 = st.columns(2)
            with c1:
                st.image(orig_img, caption="Original", use_container_width=True)
            with c2:
                st.image(grad_img, caption="Attention Map", use_container_width=True)

if __name__ == "__main__":
    main()