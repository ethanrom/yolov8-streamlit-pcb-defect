import cv2
import requests
import os
import streamlit as st
import numpy as np
from PIL import Image

from ultralytics import YOLO

model = YOLO('yolov8_pcb.pt')

def show_preds_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = model.predict(source=image)
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

st.set_page_config(page_title="PCB Defect Detector")

st.title("PCB Defect Detector")

image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image is not None:
    img = np.array(Image.open(image))
    output_image = show_preds_image(img)
    st.image(output_image, caption="Output Image", use_column_width=True)
