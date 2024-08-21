import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av

st.title('Streamlit App Test')
st.write('Gray Scale -> Color')

test_state = st.checkbox('Gray Scale -> Color ')


def callback(frame):
    img = frame.to_ndarray(format = 'bgr24')

    if test_state == True:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = av.VideoFrame.from_ndarray(img, format='gray')
    else:
        img = av.VideoFrame.from_ndarray(img, format='bgr24')

    return img

col1,col2 = st.columns(2)
with col1:
    st.write('Gray Scale')
    webrtc_streamer(key='col1', video_frame_callback=callback)
with col2:
    st.write('Color')
    webrtc_streamer(key='col2', video_frame_callback=callback)

