import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Pencil Sketch Generator")
st.write("Upload an image to create a pencil sketch effect on detected faces.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Convert the uploaded file to a numpy array and decode it with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Error: Image not found.")
    else:
        # Load the Haar cascade for face detection.
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert the image to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Sharpen the Image using Unsharp Masking
        gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)
        
        # Detect faces using the sharpened image.
        faces = face_cascade.detectMultiScale(sharpened, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            st.warning("No faces detected.")
        else:
            # Create a white canvas for the final sketch.
            canvas = 255 * np.ones_like(gray)
            
            # Process each detected face.
            for (x, y, w, h) in faces:
                # Extend the region upward to include some hair.
                hair_offset = int(0.3 * h)
                y_start = max(y - hair_offset, 0)
                roi = sharpened[y_start:y+h, x:x+w]
                
                # Step 2: Apply Bilateral Filter for Noise Reduction.
                filtered = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
                
                # Step 3: Apply Gaussian Blur to smooth edges.
                blurred = cv2.GaussianBlur(filtered, (3, 3), 0)
                
                # Step 4: Edge Detection using Canny.
                edges = cv2.Canny(blurred, threshold1=10, threshold2=50)
                
                # Step 5: Thinning Edges for a More Defined Sketch.
                try:
                    edges_thin = cv2.ximgproc.thinning(edges)  # Requires opencv-contrib-python
                except Exception as e:
                    st.info("Thinning not available, using raw edges.")
                    edges_thin = edges
                
                # Step 6: Invert the Edges for a Sketch Effect.
                sketch = 255 - edges_thin
                
                # Step 7: Remove Noise with Morphological Opening.
                kernel_open = np.ones((3, 3), np.uint8)
                sketch = cv2.morphologyEx(sketch, cv2.MORPH_OPEN, kernel_open)
                
                # Place the resulting sketch into the corresponding region on the canvas.
                canvas[y_start:y+h, x:x+w] = sketch
            
            # Resize canvas for display (enlarge by factor of 4)
            h, w = canvas.shape
            canvas_resized = cv2.resize(canvas, (w*4, h*4))
            
            # Convert original image to RGB for display.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display the Original Image and the Pencil Sketch side by side.
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption="Original Image", use_column_width=True)
            with col2:
                st.image(canvas_resized, caption="Pencil Sketch", use_column_width=True, channels="GRAY")
