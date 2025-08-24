import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import gdown
import json

# ====================
# CSS Styling Aman
# ====================
st.markdown(
"""
<style>
body { background-color: white; }

body::before {
    content: "";
    position: absolute;
    top: -30px; left: -50px;
    width: 250px; height: 250px;
    background: url("https://i.ibb.co/Lh2W1tV/leaf-top.png") no-repeat;
    background-size: contain;
    transform: rotate(20deg);
    z-index: -1;
}

body::after {
    content: "";
    position: absolute;
    bottom: -30px; right: -50px;
    width: 250px; height: 250px;
    background: url("https://i.ibb.co/Z2ShYDC/leaf-bottom.png") no-repeat;
    background-size: contain;
    transform: rotate(-15deg);
    z-index: -1;
}

.c
