import streamlit as st
import uuid

if 'unique_id' not in st.session_state:
  st.session_state["unique_id"] = []

col5, col6 = st.columns(2)

with col5:
  if st.button("追加", key=6):
    st.session_state["unique_id"].append(uuid.uuid1())

with col6:
  if st.button("削除", key=7):
    st.session_state["unique_id"].pop(-1)
    
for unique_id in st.session_state["unique_id"]:
  
  with st.container():
    col7, col8 = st.columns(2)

    with col7:
      slider_value = st.slider(
        "数値",
        min_value=0,
        max_value=14,
        value=0,
        key=unique_id
      )
    with col8:
      st.write("")
      st.write("")
      st.write(slider_value)