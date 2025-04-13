
# 🛣 Smart Route Planner with Pothole Detection using YOLO and Streamlit

This project combines object detection using YOLO with smart route planning to identify potholes (referred to as "peddles") in images, videos, webcam input, and entire directories. It also features a simulated route planner using Folium maps to help visualize potential potholes along a journey.

## 🚀 Features

- 📷 *Pothole Detection* via:
  - Uploaded Images
  - Uploaded Videos (sampled frames)
  - Real-time Webcam Stream
  - Batch Image Directory Processing

- 🗺 *Smart Route Planner*:
  - Simulated route between any two predefined cities
  - Random pothole generation along the route
  - Interactive Folium map with start, end, and pothole markers

- 🔊 *Voice Alerts*:
  - Audio feedback describing pothole positions and sizes
  - Text-to-speech responses using Google Text-to-Speech (gTTS)

- 🎯 *Customizable Detection*:
  - Confidence threshold slider
  - IOU threshold slider

## 🧠 Tech Stack

- *Python*
- *YOLOv5* via ultralytics for object detection
- *Streamlit* for interactive UI
- *Folium* for map-based route visualization
- *OpenCV* and *PIL* for image processing
- *gTTS* for text-to-speech audio generation

## 🛠 Setup Instructions

1. *Clone this repository*:
   ```bash
   git clone https://github.com/your-username/pothole-smart-route-planner.git
   cd pothole-smart-route-planner
