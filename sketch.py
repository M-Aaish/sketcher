import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the Haar cascade for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sharpen the Image using Unsharp Masking
    gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)
    
    # Detect faces in the image using the sharpened version.
    faces = face_cascade.detectMultiScale(sharpened, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        st.warning("No faces detected.")
        return None
    
    # Create a white canvas for the final sketch.
    canvas = 255 * np.ones_like(gray)
    
    for (x, y, w, h) in faces:
        hair_offset = int(0.3 * h)
        y_start = max(y - hair_offset, 0)
        roi = sharpened[y_start:y+h, x:x+w]
        
        # Apply Bilateral Filter for Noise Reduction
        filtered = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Apply Gaussian Blur to smooth edges
        blurred = cv2.GaussianBlur(filtered, (3, 3), 0)
        
        # Edge Detection using Canny
        edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
        
        # Thinning Edges for a More Defined Sketch
        try:
            edges_thin = cv2.ximgproc.thinning(edges)  # Requires opencv-contrib-python
        except Exception:
            edges_thin = edges
        
        # Invert the Edges for a Sketch Effect
        sketch = 255 - edges_thin
        
        # Remove Noise with Morphological Opening
        kernel_open = np.ones((3, 3), np.uint8)
        sketch = cv2.morphologyEx(sketch, cv2.MORPH_OPEN, kernel_open)
        
        # Place the resulting sketch into the corresponding region on the canvas.
        canvas[y_start:y+h, x:x+w] = sketch
    
    return canvas

# Streamlit UI
st.title("Face Sketch Generator")
st.write("Upload an image, and the app will generate a pencil sketch of the detected faces.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Read the image
    image = np.array(Image.open(uploaded_file))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process the image
    sketch = process_image(image)
    
    if sketch is not None:
        # Convert images for display
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(sketch, caption="Pencil Sketch", use_column_width=True, channels="GRAY")
