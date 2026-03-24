import os
import sys

print(f"Python version: {sys.version}")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    # Simple TF operation
    print(f"TF addition: {tf.add(1, 2)}")
except Exception as e:
    print(f"TensorFlow error: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV error: {e}")

try:
    import streamlit as st
    print(f"Streamlit version: {st.__version__}")
except Exception as e:
    print(f"Streamlit error: {e}")

print("Diagnostic complete.")
