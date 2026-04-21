import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from data.synthetic_generator import SyntheticDatasetGenerator

def build_3class_cnn(input_shape=(128, 128, 3)):
    """A lightweight 3-class CNN that can train quickly without a GPU."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_and_preprocess_dataset(num_samples=300):
    print("Generating Synthetic Dataset...")
    gen_dir = os.path.join(current_dir, "data", "synthetic")
    os.makedirs(gen_dir, exist_ok=True)
    gen = SyntheticDatasetGenerator(output_dir=gen_dir)
    
    X = []
    y = []
    for i in range(num_samples):
        class_label = i % 3 # Balanced dataset classes: 0, 1, 2
        sample = gen.generate_sample(f"train_{i}", class_label=class_label)
        
        mri = cv2.resize(cv2.imread(sample['mri_path'], cv2.IMREAD_GRAYSCALE), (128, 128))
        xray = cv2.resize(cv2.imread(sample['xray_path'], cv2.IMREAD_GRAYSCALE), (128, 128))
        mwi = cv2.resize(cv2.imread(sample['mwi_path'], cv2.IMREAD_GRAYSCALE), (128, 128))
        
        # Stack true multi-modal (128x128x3)
        fused = np.stack((mri, xray, mwi), axis=-1).astype("float32") / 255.0
        X.append(fused)
        y.append(class_label)
        
    return np.array(X), np.array(y)

print("Starting training process...")
X_train, y_train = load_and_preprocess_dataset(300)

model = build_3class_cnn()
print("Training model...")
# Train the model on the data
model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.2, verbose=1)

save_path = os.path.join(current_dir, "app", "brainTumor.keras")
model.save(save_path)
print(f"Model successfully saved to {save_path}")

# Generate test cases to verify
print("\n--- RUNNING TEST CASES ---")
test_gen = SyntheticDatasetGenerator(output_dir=os.path.join(current_dir, "data", "synthetic"))
class_labels = ["Normal", "Cancer", "Malformed"]

for class_idx in range(3):
    sample = test_gen.generate_sample(f"test_{class_idx}", class_label=class_idx)
    mri = cv2.resize(cv2.imread(sample['mri_path'], cv2.IMREAD_GRAYSCALE), (128, 128))
    xray = cv2.resize(cv2.imread(sample['xray_path'], cv2.IMREAD_GRAYSCALE), (128, 128))
    mwi = cv2.resize(cv2.imread(sample['mwi_path'], cv2.IMREAD_GRAYSCALE), (128, 128))
    
    fused = np.expand_dims(np.stack((mri, xray, mwi), axis=-1).astype("float32") / 255.0, axis=0)
    preds = model.predict(fused, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    
    print(f"True Class: {class_labels[class_idx]} | Predicted: {class_labels[pred_idx]} | Softmax: {[round(float(p), 4) for p in preds]}")
    assert class_idx == pred_idx, "Model prediction failed!"
print("All test cases passed perfectly.")
