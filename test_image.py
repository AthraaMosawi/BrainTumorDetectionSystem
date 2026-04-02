import cv2
import numpy as np
import os
from PIL import Image

def test_img(img_name):
    path = f"/Users/athraamosawi/Documents/Hassan/tumor-detection-master/sample/{img_name}"
    img = Image.open(path).convert('L')
    gray = np.array(img)
    
    # Same as enhance_image
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    max_int = int(np.max(enhanced))
    print(f"{img_name} max intensity: {max_int}")
    
    # How many pixels > 240, 225, 200
    for thresh in [240, 225, 200, 180, 150]:
        px = int(np.sum(enhanced > thresh))
        print(f"Threshold {thresh}: {px} pixels")

test_img('mri5.jpg')
test_img('xray.jpg')
