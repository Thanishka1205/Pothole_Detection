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
import folium
from streamlit_folium import folium_static
import random

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

# Initialize global model
model = None

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'show_login' not in st.session_state:
    st.session_state.show_login = False
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Mock user database (replace with secure database in production)
USER_CREDENTIALS = {
    "admin": "password123",
    "user1": "test456"
}

# Audio functions
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = f"temp_audio_{time.time()}.mp3"
    tts.save(audio_file)
    return audio_file

def autoplay_audio(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)

# Route Simulation Functions
def generate_route(start_coords, end_coords):
    """Generate a simulated route with random potholes"""
    num_points = 20
    lat_step = (end_coords[0] - start_coords[0]) / num_points
    lng_step = (end_coords[1] - start_coords[1]) / num_points
    
    route = []
    for i in range(num_points + 1):
        route.append((
            start_coords[0] + i * lat_step,
            start_coords[1] + i * lng_step
        ))
    
    route = [(lat + random.uniform(-0.001, 0.001), 
              lng + random.uniform(-0.001, 0.001)) for lat, lng in route]
    
    return route

def simulate_potholes(route):
    """Add simulated potholes along the route"""
    potholes = []
    for i, point in enumerate(route):
        if random.random() > 0.6 * (i/len(route)):
            potholes.append((point[0] + random.uniform(-0.0005, 0.0005),
                            point[1] + random.uniform(-0.0005, 0.0005)))
    return potholes

def show_route_map(start_coords, end_coords, route=None, potholes=None, view_only=False):
    """Display the route using Folium/OpenStreetMap"""
    m = folium.Map(
        location=[(start_coords[0] + end_coords[0])/2, 
                 (start_coords[1] + end_coords[1])/2],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    folium.Marker(
        start_coords,
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
        popup='Start'
    ).add_to(m)
    
    folium.Marker(
        end_coords,
        icon=folium.Icon(color='red', icon='stop', prefix='fa'),
        popup='Destination'
    ).add_to(m)
    
    if not view_only and route and potholes:
        folium.PolyLine(
            route,
            color='blue',
            weight=5,
            opacity=0.7,
            popup='Suggested Route'
        ).add_to(m)
        
        for lat, lng in potholes:
            folium.CircleMarker(
                location=[lat, lng],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                popup='Pothole'
            ).add_to(m)
    
    folium_static(m, width=800, height=500)

def get_coordinates(location_name):
    """Mock geocoding - in a real app, use Nominatim or similar"""
    mock_locations = {
        "chennai": (13.0827, 80.2707),
        "coimbatore": (11.0168, 76.9558),
        "madurai": (9.9252, 78.1198),
        "tiruchirappalli": (10.7905, 78.7047),
        "salem": (11.6643, 78.1460),
        "erode": (11.3410, 77.7172)
    }
    return mock_locations.get(location_name.lower(), (0, 0))

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

@st.cache_resource
def load_model(path):
    try:
        if not path.exists():
            st.warning(f"Model file not found at: {path}")
            return None
        st.info(f"Loading model from: {path}")
        model_instance = YOLO(str(path))
        st.success("Model loaded successfully!")
        return model_instance
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def is_pothole_detection_valid(results):
    return len(results[0].boxes) > 0

def initialize_model():
    global model
    DEFAULT_MODEL_PATH = Path("models/best_pothole_model.pt")
    if model is None:
        if DEFAULT_MODEL_PATH.exists():
            model = load_model(DEFAULT_MODEL_PATH)
        else:
            st.warning("Default model not found at models/best_pothole_model.pt")
            uploaded_model = st.file_uploader("Upload a .pt model", type=["pt"], key="model_uploader")
            if uploaded_model is not None:
                temp_model_path = Path(tempfile.mktemp(suffix=".pt"))
                try:
                    with open(temp_model_path, "wb") as f:
                        f.write(uploaded_model.read())
                    model = load_model(temp_model_path)
                finally:
                    if os.path.exists(temp_model_path):
                        os.unlink(temp_model_path)
    return model

def process_directory(directory_path, model, confidence_threshold, iou_threshold):
    if model is None:
        st.error("Please load a valid model first.")
        return
    
    st.subheader(f"Processing Directory: {directory_path}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(directory_path) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        st.warning("No image files found in the directory")
        return
    
    results_container = st.container()
    
    total_potholes = 0
    processed_images = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(directory_path, image_file)
        
        try:
            image = Image.open(image_path)
            img_array = np.array(image)
            
            results = model.predict(
                source=img_array,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            
            num_potholes = len(results[0].boxes)
            total_potholes += num_potholes
            
            with results_container:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption=f"Original: {image_file}", use_column_width=True)
                with col2:
                    if num_potholes > 0:
                        boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]
                        plotted_img = plot_boxes(img_array, boxes)
                        st.image(plotted_img, 
                                caption=f"Detected {num_potholes} pothole(s)", 
                                use_column_width=True)
                    else:
                        st.image(image, caption="No potholes detected", use_column_width=True)
                
                st.markdown("---")
            
            processed_images += 1
            
            progress = (i + 1) / len(image_files)
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1}/{len(image_files)} images ({progress:.1%})")
            
        except Exception as e:
            st.error(f"Error processing {image_file}: {str(e)}")
    
    st.success(f"Finished processing {processed_images} images. Total potholes detected: {total_potholes}")
    
    if total_potholes > 0:
        announcement = f"Found {total_potholes} potholes in {processed_images} images"
    else:
        announcement = f"No potholes detected in {processed_images} images"
    
    audio_file = text_to_speech(announcement)
    autoplay_audio(audio_file)

def process_webcam(model, confidence_threshold, iou_threshold):
    if model is None:
        st.error("Please load a valid model first.")
        return
    
    st.subheader("Live Pothole Detection")
    
    # Create a placeholder for the webcam feed
    FRAME_WINDOW = st.image([])
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button and not st.session_state.webcam_active:
        st.session_state.webcam_active = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam")
            st.session_state.webcam_active = False
            return
        
        last_announcement_time = 0
        announcement_cooldown = 3
        last_detection_time = 0
        no_pothole_cooldown = 5
        
        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(
                source=frame_rgb,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            
            current_time = time.time()
            valid_potholes = is_pothole_detection_valid(results)
            
            if valid_potholes:
                boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]
                annotated_frame = plot_boxes(frame_rgb, boxes)
                FRAME_WINDOW.image(annotated_frame)
                last_detection_time = current_time
            else:
                FRAME_WINDOW.image(frame_rgb)
                
                if (current_time - last_detection_time) > no_pothole_cooldown and \
                   (current_time - last_announcement_time) > announcement_cooldown:
                    audio_file = text_to_speech("No pothole found")
                    autoplay_audio(audio_file)
                    last_announcement_time = current_time
            
            if valid_potholes and (current_time - last_announcement_time) > announcement_cooldown:
                img_width = frame.shape[1]
                img_area = frame.shape[0] * frame.shape[1]
                
                announcements = []
                for box in results[0].boxes:
                    box_coords = box.xyxy[0].cpu().numpy()
                    x_center = (box_coords[0] + box_coords[2]) / 2
                    direction = get_direction(x_center, img_width)
                    size_category = get_size_category(box_coords, img_area)
                    announcements.append(f"{size_category} pothole on the {direction}")
                
                if len(announcements) == 1:
                    announcement = announcements[0]
                else:
                    unique_announcements = list(set(announcements))
                    if len(unique_announcements) == 1:
                        count = len(announcements)
                        announcement = f"{count} {unique_announcements[0].replace('pothole', 'potholes')}"
                    else:
                        announcement = "Multiple potholes detected in various locations"
                
                audio_file = text_to_speech(announcement)
                autoplay_audio(audio_file)
                last_announcement_time = current_time
                last_detection_time = current_time
            
            # Check for stop condition
            if stop_button:
                st.session_state.webcam_active = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
        st.session_state.webcam_active = False
        st.rerun()

def process_image(uploaded_file, model, confidence_threshold, iou_threshold):
    if model is None:
        st.error("Please load a valid model first.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detection Results")
        with st.spinner("Detecting potholes..."):
            img_array = np.array(image)
            results = model.predict(
                source=img_array,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            
            boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]
            plotted_img = plot_boxes(img_array, boxes)
            st.image(plotted_img, use_container_width=True, channels="RGB")
            
            num_potholes = len(results[0].boxes)
            if num_potholes > 0:
                img_width = img_array.shape[1]
                img_area = img_array.shape[0] * img_array.shape[1]
                
                announcements = []
                for box in results[0].boxes:
                    box_coords = box.xyxy[0].cpu().numpy()
                    x_center = (box_coords[0] + box_coords[2]) / 2
                    direction = get_direction(x_center, img_width)
                    size_category = get_size_category(box_coords, img_area)
                    announcements.append(f"{size_category} pothole on the {direction}")
                
                if len(announcements) == 1:
                    announcement = announcements[0]
                else:
                    unique_announcements = list(set(announcements))
                    if len(unique_announcements) == 1:
                        count = len(announcements)
                        announcement = f"{count} {unique_announcements[0].replace('pothole', 'potholes')}"
                    else:
                        announcement = "Multiple potholes detected in various locations"
                
                st.success("Potholes detected!")
                audio_file = text_to_speech(announcement)
                autoplay_audio(audio_file)
            else:
                st.success("No potholes detected")
                audio_file = text_to_speech("No potholes detected")
                autoplay_audio(audio_file)

def process_video(uploaded_file, model, confidence_threshold, iou_threshold):
    if model is None:
        st.error("Please load a valid model first.")
        return
    
    st.warning("Video processing may take some time. Please be patient.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    
    st.subheader("Original Video")
    st.video(temp_path)
    
    if st.button("Analyze Video for Potholes"):
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = width * height
        
        frame_interval = max(1, total_frames // 5)
        sampled_frames = []
        frame_numbers = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(
                source=frame_rgb,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            
            boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]
            annotated_frame = plot_boxes(frame_rgb, boxes)
            sampled_frames.append(annotated_frame)
            frame_numbers.append(i)
            
            progress = min((i + frame_interval) / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {i}/{total_frames} ({progress:.1%})")
        
        cap.release()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        st.subheader("Sampled Frames with Pothole Detections")
        
        total_potholes = 0
        for frame_num, frame in zip(frame_numbers, sampled_frames):
            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=640
            )
            num_potholes = len(results[0].boxes)
            total_potholes += num_potholes
            
            st.image(frame, channels="RGB", caption=f"Frame {frame_num}: {num_potholes} pothole(s)")
        
        if total_potholes > 0:
            announcement = f"Found {total_potholes} potholes across {len(sampled_frames)} sampled frames"
        else:
            announcement = "No potholes detected in sampled frames"
        
        st.success(announcement)
        audio_file = text_to_speech(announcement)
        autoplay_audio(audio_file)

def process_uploaded_file(uploaded_file, model, confidence_threshold, iou_threshold):
    if model is None:
        st.error("Please load a valid model first.")
        return
    file_type = uploaded_file.type.split('/')[0]
    if file_type == "image":
        process_image(uploaded_file, model, confidence_threshold, iou_threshold)
    elif file_type == "video":
        process_video(uploaded_file, model, confidence_threshold, iou_threshold)
    else:
        st.error("Unsupported file type. Please upload an image or video.")

def login_page():
    st.set_page_config(
        page_title="Login - Route Planning",
        page_icon="🔒",
        layout="centered"
    )
    
    st.title("🔒 Login for Route Planning")
    st.markdown("Please sign in to plan routes with pothole detection")
    
    with st.form(key="login_form"):
        username = st.text_input("Username", placeholder="Enter your username", key="login_username")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
        submit_button = st.form_submit_button("Login")
    
    if submit_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.show_login = False
            st.success(f"Welcome, {username}! Redirecting to the app...")
            time.sleep(2)
            st.rerun()
        else:
            st.error("Invalid username or password.")

def main_app():
    global model
    st.set_page_config(
        page_title="Pothole Detection System",
        page_icon="🕳",
        layout="wide"
    )

    st.title("🛣 Pothole Detection and Route Planner")
    
    # Initialize model
    model = initialize_model()

    # Sidebar
    with st.sidebar:
        st.header("📍 Route Planning")
        start_loc = st.text_input("Start Location", "Salem", key="start_loc_input")
        end_loc = st.text_input("Destination", "Coimbatore", key="end_loc_input")
        
        start_coords = get_coordinates(start_loc)
        end_coords = get_coordinates(end_loc)
        
        if start_coords == (0, 0) or end_coords == (0, 0):
            st.warning("Please enter valid locations")
        else:
            st.subheader("Map Preview")
            show_route_map(start_coords, end_coords, view_only=True)
        
        if st.button("Plan Optimal Route", key="route_button"):
            if not st.session_state.logged_in:
                st.session_state.show_login = True
                st.rerun()
            else:
                if start_coords == (0, 0) or end_coords == (0, 0):
                    st.error("Please enter valid locations")
                else:
                    with st.spinner("Finding the best route..."):
                        route = generate_route(start_coords, end_coords)
                        potholes = simulate_potholes(route)
                        st.success("Route found! Follow the blue path.")
                        show_route_map(start_coords, end_coords, route, potholes)
                        audio_file = text_to_speech("Follow the suggested route to reach your destination easily")
                        autoplay_audio(audio_file)
        
        if not st.session_state.logged_in:
            st.markdown("Sign up or login to find the best route")
            if st.button("Login", key="sidebar_login_button"):
                st.session_state.show_login = True
                st.rerun()
        else:
            st.header(f"👤 {st.session_state.username}")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.show_login = False
                st.rerun()

        st.header("⚙ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.25, 0.01,
            key="confidence_slider"
        )
        iou_threshold = st.slider(
            "IOU Threshold", 
            0.0, 1.0, 0.45, 0.01,
            key="iou_slider"
        )

    # Main content
    st.header("🔍 Pothole Detection")
    
    center_container = st.container()
    with center_container:
        cols = st.columns([1, 3, 1])
        
        with cols[1]:
            detection_mode = st.radio(
                "Select input type:",
                ("Image", "Video", "Webcam", "Directory"),
                horizontal=True,
                key="detection_mode_radio"
            )
            
            st.markdown("---")
            
            if detection_mode in ["Image", "Video"]:
                uploaded_file = st.file_uploader(
                    f"Upload an {detection_mode.lower()}",
                    type=["jpg", "jpeg", "png"] if detection_mode == "Image" else ["mp4", "mov"],
                    key=f"{detection_mode.lower()}_uploader"
                )
                if uploaded_file is not None:
                    process_uploaded_file(uploaded_file, model, confidence_threshold, iou_threshold)
            
            elif detection_mode == "Directory":
                directory_path = st.text_input(
                    "Enter directory path containing images:",
                    key="directory_path_input"
                )
                if directory_path and os.path.isdir(directory_path):
                    if st.button("Process Directory", key="process_dir_button"):
                        process_directory(directory_path, model, confidence_threshold, iou_threshold)
                else:
                    st.warning("Please enter a valid directory path")
            
            elif detection_mode == "Webcam":
                process_webcam(model, confidence_threshold, iou_threshold)

def main():
    if st.session_state.show_login:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
