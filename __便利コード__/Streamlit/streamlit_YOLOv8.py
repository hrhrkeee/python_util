import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
from ultralytics import YOLO
from logging import getLogger

logger = getLogger('ultralytics')
logger.disabled = True

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8x.pt")
# model = YOLO("yolov8x-seg.pt")

st.title('Streamlit YOLOv8 Test')

def callback(frame):
    img = frame.to_ndarray(format = 'rgb24')

    img = model(img, conf = 0.60)[0].plot()

    img = av.VideoFrame.from_ndarray(img, format='rgb24')

    return img

webrtc_streamer(key='col1', video_frame_callback=callback)

