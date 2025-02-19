import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import FaceRecognitionInference

# Streamlit Page Configuration
st.set_page_config(page_title="🔍 Face Recognition App", layout="wide")

# Load Face Recognition Model
MODEL_PATH = r"..\models\face_recognition.onnx"

# Toggle similarity method
use_similarity = st.sidebar.checkbox("Use Similarity-Based Recognition", value=False)

# Initialize the Face Recognition System
face_recognizer = FaceRecognitionInference(
    model_path=MODEL_PATH,
    model_type="onnx",
    use_similarity=use_similarity,
)

# Streamlit App Title
st.title("🔍 Face Recognition System")
st.write("Upload an image or use your webcam to detect and recognize faces.")

# Option to select input method
input_method = st.radio("Select input method:", ["Upload Image", "Use Webcam"])

image = None  # Initialize image variable

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))  # Convert to NumPy array
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

elif input_method == "Use Webcam":
    webcam_image = st.camera_input("Take a picture")
    
    if webcam_image is not None:
        image = np.array(Image.open(webcam_image))  # Convert to NumPy array
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

if image is not None:
    # Perform Face Detection & Recognition
    predictions, output_img = face_recognizer.detect_and_recognize(image)

    # Display Processed Image in the center
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_container_width=True)

    # Display Predictions
    if predictions:
        st.success("Recognized Faces:")
        for name in predictions:
            st.write(f"🧑 {name}")
    else:
        st.warning("🚨 No faces recognized.")

else:
    st.warning("⚠️ Please upload an image or capture one using the webcam.")
