import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
from ultralytics import SAM
from ultralytics import FastSAM
from logging import getLogger

logger = getLogger('ultralytics')
logger.disabled = True

# model = SAM("sam2_t.pt")
model = FastSAM("FastSAM-x.pt")

st.title('Streamlit YOLOv8 Test!')

def callback(frame):
    img = frame.to_ndarray(format = 'rgb24')

    img = model(img, conf = 0.40)[0].plot()

    img = av.VideoFrame.from_ndarray(img, format='rgb24')

    return img

webrtc_streamer(key='col1', video_frame_callback=callback)

