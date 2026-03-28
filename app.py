import gradio as gr
from api.predict import predict_image
import io

def predict_fn(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    result = predict_image(buf.read())
    return result['prediction']

gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type="pil", label="Upload Colonoscopy Image"),
    outputs=gr.Textbox(label="Disease Predicted"),
    title="Automated Multi-Class Gastrointestinal Anomaly Detection in Colonoscopy Imagery",
    description="Upload a colonoscopy image"
).launch()
