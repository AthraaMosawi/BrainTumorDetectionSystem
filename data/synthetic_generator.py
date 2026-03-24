import numpy as np
import cv2
import os
import random

class SyntheticDatasetGenerator:
    """
    Generates synthetic multi-modal datasets (X-ray, MRI, MWI) 
    where a tumor is placed at the same spatial coordinates.
    """
    def __init__(self, output_dir=None):
        if output_dir is None:
            # Get the path to the 'data/synthetic' directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, 'synthetic')
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_sample(self, sample_id, tumor_pos=None, tumor_radius=None):
        if tumor_pos is None:
            tumor_pos = (random.randint(30, 98), random.randint(30, 98))
        if tumor_radius is None:
            tumor_radius = random.randint(4, 10)
            
        # 1. Generate MRI-like image (Soft tissue)
        mri = np.random.randint(5, 15, (128, 128), dtype=np.uint8) # Background noise
        cv2.circle(mri, (64, 64), 50, 40, -1) # "Brain" phantom
        cv2.circle(mri, tumor_pos, tumor_radius, 220, -1) # Tumor (bright in MRI T2/FLAIR)
        mri = cv2.GaussianBlur(mri, (3, 3), 0)

        # 2. Generate X-ray-like image (Anatomical/Bone structure)
        xray = np.random.randint(2, 8, (128, 128), dtype=np.uint8) # Background noise
        cv2.rectangle(xray, (20, 20), (108, 108), 80, 2) # "Skull" phantom boundary
        cv2.circle(xray, tumor_pos, tumor_radius, 180, -1) # Tumor (dense area)
        xray = cv2.GaussianBlur(xray, (3, 3), 0)

        # Save images
        mri_path = os.path.join(self.output_dir, f"sample_{sample_id}_mri.png")
        xray_path = os.path.join(self.output_dir, f"sample_{sample_id}_xray.png")
        cv2.imwrite(mri_path, mri)
        cv2.imwrite(xray_path, xray)
        
        return {
            'id': sample_id,
            'tumor_pos': tumor_pos,
            'tumor_radius': tumor_radius,
            'mri_path': mri_path,
            'xray_path': xray_path
        }

    def generate_batch(self, count=10):
        samples = []
        for i in range(count):
            samples.append(self.generate_sample(i))
        return samples

if __name__ == "__main__":
    gen = SyntheticDatasetGenerator()
    samples = gen.generate_batch(10)
    print(f"Successfully generated {len(samples)} synthetic multi-modal samples in d:/m.jawad/project/xray/data/synthetic")
