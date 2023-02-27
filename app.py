import streamlit as st
import cv2
import requests
import os
import numpy as np

from ultralytics import YOLO

model = YOLO('yolov8_pcb.pt')
path = [['spur.jpg'], ['mouse.jpg']]

def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()
    for i, det in enumerate(results.boxes.xyxy):
        cv2.rectangle(
            image,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

st.title("PCB Defect Detector")

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    output_image = show_preds_image(uploaded_file)
    st.image(output_image, caption='Output Image.', use_column_width=True)

    if st.button("Save Image"):
        cv2.imwrite("output_image.jpg", output_image)
        st.success("Image saved successfully!")
