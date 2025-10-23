import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="YOLOv5 Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 YOLOv5 Object Detection Web App")
st.markdown("""Upload an image or video to detect objects using YOLOv5 deep learning model.""")

# Sidebar configuration
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
model_size = st.sidebar.selectbox("Model Size", ["yolov5s", "yolov5m", "yolov5l", "yolov5x"])

# Load YOLOv5 model
@st.cache_resource
def load_model(model_name):
    """Load YOLOv5 model from torch hub"""
    try:
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the selected model
with st.spinner(f"Loading {model_size} model..."):
    model = load_model(model_size)
    if model:
        model.conf = confidence
        st.sidebar.success(f"Model {model_size} loaded successfully!")

# File upload
upload_type = st.radio("Select input type:", ["Image", "Video"])

if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and model is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Perform detection
        with st.spinner("Detecting objects..."):
            results = model(img_array)
            
            # Get detection results
            detections = results.pandas().xyxy[0]
            
            # Render results on image
            result_img = np.squeeze(results.render())
            
        with col2:
            st.subheader("Detection Results")
            st.image(result_img, use_container_width=True)
        
        # Display detection details
        st.subheader("Detected Objects")
        if len(detections) > 0:
            st.dataframe(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
            
            # Object count summary
            object_counts = detections['name'].value_counts()
            st.subheader("Object Count Summary")
            st.bar_chart(object_counts)
        else:
            st.info("No objects detected. Try lowering the confidence threshold.")

elif upload_type == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None and model is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Open video
        cap = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Process video button
        if st.button("Process Video"):
            # Create temporary output file
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform detection
                results = model(frame)
                result_frame = np.squeeze(results.render())
                
                # Convert RGB to BGR for video writing
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                out.write(result_frame_bgr)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Release resources
            cap.release()
            out.release()
            
            # Display processed video
            st.success("Video processing complete!")
            st.video(output_file.name)
            
            # Cleanup
            os.unlink(tfile.name)
        else:
            # Display first frame as preview
            ret, frame = cap.read()
            if ret:
                st.subheader("Video Preview (First Frame)")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()
            os.unlink(tfile.name)

# Information section
with st.expander("ℹ️ About YOLOv5"):
    st.markdown("""
    **YOLO (You Only Look Once)** is a state-of-the-art, real-time object detection system.
    
    **YOLOv5 Models:**
    - **yolov5s**: Smallest, fastest (less accurate)
    - **yolov5m**: Medium size and speed
    - **yolov5l**: Large (more accurate)
    - **yolov5x**: Largest, most accurate (slower)
    
    **Detected Classes:** YOLOv5 can detect 80 different object classes from the COCO dataset,
    including people, vehicles, animals, and common objects.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and YOLOv5 🚀")
