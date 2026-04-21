import os
import sys
import numpy as np
import cv2
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from data.synthetic_generator import SyntheticDatasetGenerator

print("Loading model...")
model_path = os.path.join(current_dir, "app", "brainTumor.keras")
if not os.path.exists(model_path):
    model_path = os.path.join(current_dir, "brainTumor.keras")
model = tf.keras.models.load_model(model_path)
print(f"Model output shape expected: {model.output_shape}")

# Ask the model what its input shape is
input_shape = model.input_shape
print(f"Model input shape expected: {input_shape}")
# Input shape might be (None, 128, 128, 3) or (None, 224, 224, 3)
target_size = input_shape[1:3] # (128, 128)

print("\nGenerating Synthetic Data...")
gen = SyntheticDatasetGenerator(output_dir=os.path.join(current_dir, "tmp_test"))
class_labels = ["Normal", "Cancer", "Malformed"]

for class_idx in range(3):
    sample = gen.generate_sample(f"debug_{class_idx}", class_label=class_idx)
    
    # Read the generated grayscale images
    mri_gray = cv2.imread(sample['mri_path'], cv2.IMREAD_GRAYSCALE)
    xray_gray = cv2.imread(sample['xray_path'], cv2.IMREAD_GRAYSCALE)
    mwi_gray = cv2.imread(sample['mwi_path'], cv2.IMREAD_GRAYSCALE)
    
    # In dashboard.py we stack them like this:
    mri_reshaped = cv2.resize(mri_gray, target_size)
    xray_reshaped = cv2.resize(xray_gray, target_size)
    mwi_reshaped = cv2.resize(mwi_gray, target_size)
    
    fused_tensor = np.stack((mri_reshaped, xray_reshaped, mwi_reshaped), axis=-1)
    
    # Normalize
    img_in = (fused_tensor.astype("float32") / 255.0)
    img_in = np.expand_dims(img_in, axis=0) # Add batch dim
    
    preds = model.predict(img_in, verbose=0)[0]
    pred_idx = np.argmax(preds)
    
    print(f"--- TEST CASE: True Class = {class_labels[class_idx]} ---")
    print(f"Raw Softmax Output: {preds}")
    print(f"Predicted Class: {class_labels[pred_idx]}")
    print(f"Is it correct? {class_idx == pred_idx}\n")

