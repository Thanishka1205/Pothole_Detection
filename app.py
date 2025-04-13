import os
import sys
import warnings
import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path
from gtts import gTTS
import base64
import time
import math
from collections import defaultdict

# ============================================
# YOLO IMPORT WORKAROUND FOR STREAMLIT
# ============================================
os.environ['YOLO_DISABLE_HUB'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)

if 'signal' in sys.modules:
    del sys.modules['signal']
import signal
from unittest.mock import MagicMock
signal.signal = MagicMock()

from ultralytics import YOLO
# ============================================

# Audio functions
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = f"temp_audio_{time.time()}.mp3"
    tts.save(audio_file)
    return audio_file

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    os.unlink(file_path)

# Analysis functions
def get_direction(x_center, image_width):
    if x_center < image_width / 3:
        return "left"
    elif x_center > 2 * image_width / 3:
        return "right"
    else:
        return "center"

def get_size_category(box, image_area):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    relative_size = box_area / image_area
    return "large" if relative_size > 0.05 else "small"

def plot_boxes(image, boxes):
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

# Model loading
@st.cache_resource
def load_model(path):
    try:
        if not path.exists():
            st.warning(f"Model file not found at: {path}")
            return None
        st.info(f"Loading model from: {path}")
        model = YOLO(str(path))
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def is_peddle_detection_valid(results):
    """Check if detected objects are actually peddles (not faces/hands/etc)"""
    # Add your custom logic here based on your model's class IDs
    # For now, we'll assume all detections are peddles (modify as needed)
    return len(results[0].boxes) > 0

# Webcam processing
def process_webcam(model, confidence_threshold, iou_threshold):
    st.subheader("Live Peddle Detection")
    
    if st.button('Start Webcam'):
        with st.spinner("Initializing camera (5 second warm-up)..."):
            time.sleep(5)  # 5-second warm-up period
        
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        
        last_announcement_time = 0
        announcement_cooldown = 3  # seconds
        last_detection_time = 0
        no_peddle_cooldown = 5  # seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            # Convert and predict
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(
                source=frame_rgb,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            
            current_time = time.time()
            valid_peddles = is_peddle_detection_valid(results)
            
            # Draw boxes only if valid peddles detected
            if valid_peddles:
                boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]
                annotated_frame = plot_boxes(frame_rgb, boxes)
                FRAME_WINDOW.image(annotated_frame)
                last_detection_time = current_time
            else:
                FRAME_WINDOW.image(frame_rgb)
                
                # Announce "no peddle" if nothing detected for cooldown period
                if (current_time - last_detection_time) > no_peddle_cooldown and \
                   (current_time - last_announcement_time) > announcement_cooldown:
                    audio_file = text_to_speech("No peddle found")
                    autoplay_audio(audio_file)
                    last_announcement_time = current_time
            
            # Audio announcements for valid peddles
            if valid_peddles and (current_time - last_announcement_time) > announcement_cooldown:
                img_width = frame.shape[1]
                img_area = frame.shape[0] * frame.shape[1]
                
                announcements = []
                for box in results[0].boxes:
                    box_coords = box.xyxy[0].cpu().numpy()
                    x_center = (box_coords[0] + box_coords[2]) / 2
                    direction = get_direction(x_center, img_width)
                    size_category = get_size_category(box_coords, img_area)
                    announcements.append(f"{size_category} peddle on the {direction}")
                
                if len(announcements) == 1:
                    announcement = announcements[0]
                else:
                    unique_announcements = list(set(announcements))
                    if len(unique_announcements) == 1:
                        count = len(announcements)
                        announcement = f"{count} {unique_announcements[0].replace('peddle', 'peddles')}"
                    else:
                        announcement = "Multiple peddles detected in various locations"
                
                audio_file = text_to_speech(announcement)
                autoplay_audio(audio_file)
                last_announcement_time = current_time
                last_detection_time = current_time
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.1)
        
        cap.release()


# Processing functions
def process_image(uploaded_file, model, confidence_threshold, iou_threshold):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Detection Results")
        with st.spinner("Detecting peddles..."):
            img_array = np.array(image)
            results = model.predict(
                source=img_array,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            
            boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]
            plotted_img = plot_boxes(img_array, boxes)
            st.image(plotted_img, use_column_width=True, channels="BGR")
            
            num_peddles = len(results[0].boxes)
            if num_peddles > 0:
                img_width = img_array.shape[1]
                img_area = img_array.shape[0] * img_array.shape[1]
                
                announcements = []
                for box in results[0].boxes:
                    box_coords = box.xyxy[0].cpu().numpy()
                    x_center = (box_coords[0] + box_coords[2]) / 2
                    direction = get_direction(x_center, img_width)
                    size_category = get_size_category(box_coords, img_area)
                    announcements.append(f"{size_category} peddle on the {direction}")
                
                if len(announcements) == 1:
                    announcement = announcements[0]
                else:
                    unique_announcements = list(set(announcements))
                    if len(unique_announcements) == 1:
                        count = len(announcements)
                        announcement = f"{count} {unique_announcements[0].replace('peddle', 'peddles')}"
                    else:
                        announcement = "Multiple peddles detected in various locations"
                
                st.success("Peddles detected!")
                audio_file = text_to_speech(announcement)
                autoplay_audio(audio_file)
            else:
                st.success("No peddles detected")
                audio_file = text_to_speech("No peddles detected")
                autoplay_audio(audio_file)

def process_video(uploaded_file, model, confidence_threshold, iou_threshold):
    st.warning("Video processing may take some time. Please be patient.")
    
    # Store video file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    
    st.subheader("Original Video")
    st.video(temp_path)
    
    if st.button("Analyze Video for Peddles"):
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = width * height
        
        # Calculate frame interval to sample ~5 frames
        frame_interval = max(1, total_frames // 5)
        sampled_frames = []
        frame_numbers = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process sampled frames
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert and predict
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(
                source=frame_rgb,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            
            # Store results
            boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]
            annotated_frame = plot_boxes(frame_rgb, boxes)
            sampled_frames.append(annotated_frame)
            frame_numbers.append(i)
            
            # Update progress
            progress = min((i + frame_interval) / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {i}/{total_frames} ({progress:.1%})")
        
        cap.release()
        os.unlink(temp_path)
        
        # Display results
        st.subheader("Sampled Frames with Peddle Detections")
        
        total_peddles = 0
        for frame_num, frame in zip(frame_numbers, sampled_frames):
            # Count peddles in this frame
            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            num_peddles = len(results[0].boxes)
            total_peddles += num_peddles
            
            st.image(frame, channels="RGB", caption=f"Frame {frame_num}: {num_peddles} peddle(s)")
        
        # Final announcement
        if total_peddles > 0:
            announcement = f"Found {total_peddles} peddles across {len(sampled_frames)} sampled frames"
        else:
            announcement = "No peddles detected in sampled frames"
        
        st.success(announcement)
        audio_file = text_to_speech(announcement)
        autoplay_audio(audio_file)

def process_uploaded_file(uploaded_file, model, confidence_threshold, iou_threshold):
    file_type = uploaded_file.type.split('/')[0]
    if file_type == "image":
        process_image(uploaded_file, model, confidence_threshold, iou_threshold)
    elif file_type == "video":
        process_video(uploaded_file, model, confidence_threshold, iou_threshold)
    else:
        st.error("Unsupported file type. Please upload an image or video.")
def main():
    st.set_page_config(
        page_title="Peddle Detection System",
        page_icon="🕳️",
        layout="wide"
    )

    st.title("🕳️ Peddle Detection System")

    # Model loading - KEEPING ORIGINAL MODEL NAME
    DEFAULT_MODEL_PATH = Path("models/best_pothole_model.pt")
    model = None
    if DEFAULT_MODEL_PATH.exists():
        model = load_model(DEFAULT_MODEL_PATH)
    else:
        st.sidebar.warning("Default model not found.")
        uploaded_model = st.sidebar.file_uploader("Upload a .pt model", type=["pt"])
        if uploaded_model is not None:
            temp_model_path = Path(tempfile.mktemp(suffix=".pt"))
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.read())
            model = load_model(temp_model_path)

    # Sidebar options
    with st.sidebar:
        st.header("Detection Mode")
        detection_mode = st.radio(
            "Select input source:",
            ("File Upload", "Webcam")
        )
        
        st.header("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.01)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This system detects peddles while ignoring human faces/hands.")

    if model is None:
        st.error("Please load or upload a valid model first.")
        return

    if detection_mode == "File Upload":
        uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file, model, confidence_threshold, iou_threshold)
    else:
        process_webcam(model, confidence_threshold, iou_threshold)

if __name__ == "__main__":
    main()