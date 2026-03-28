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
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, 'synthetic')

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_sample(self, sample_id, tumor_pos=None, tumor_radius=None):
        if tumor_pos is None:
            tumor_pos = (random.randint(35, 93), random.randint(35, 93))
        if tumor_radius is None:
            tumor_radius = random.randint(7, 14)

        # 1. Generate MRI-like image (Soft tissue - T2 weighted)
        mri = np.random.randint(5, 15, (128, 128), dtype=np.uint8)
        cv2.circle(mri, (64, 64), 50, 40, -1)         # Brain phantom
        cv2.circle(mri, tumor_pos, tumor_radius, 240, -1)  # Tumor (hyper-intense, >225)
        mri = cv2.GaussianBlur(mri, (3, 3), 0)

        # 2. Generate X-ray-like image (Anatomical/Bone structure)
        xray = np.random.randint(2, 8, (128, 128), dtype=np.uint8)
        cv2.rectangle(xray, (20, 20), (108, 108), 80, 2)   # Skull boundary
        cv2.circle(xray, tumor_pos, tumor_radius, 200, -1)  # Tumor mass
        xray = cv2.GaussianBlur(xray, (3, 3), 0)

        # 3. Generate MWI-like dielectric map
        mwi = np.random.randint(2, 10, (128, 128), dtype=np.uint8)
        cv2.circle(mwi, (64, 64), 50, 30, -1)              # Brain background
        cv2.circle(mwi, tumor_pos, tumor_radius + 3, 230, -1)  # Tumor hotspot (dielectric peak)
        mwi = cv2.GaussianBlur(mwi, (5, 5), 0)

        # Save images
        mri_path  = os.path.join(self.output_dir, f"sample_{sample_id}_mri.png")
        xray_path = os.path.join(self.output_dir, f"sample_{sample_id}_xray.png")
        mwi_path  = os.path.join(self.output_dir, f"sample_{sample_id}_mwi.png")
        cv2.imwrite(mri_path, mri)
        cv2.imwrite(xray_path, xray)
        cv2.imwrite(mwi_path, mwi)

        return {
            'id': sample_id,
            'tumor_pos': tumor_pos,
            'tumor_radius': tumor_radius,
            'mri_path': mri_path,
            'xray_path': xray_path,
            'mwi_path': mwi_path,
        }

    def generate_batch(self, count=10):
        samples = []
        for i in range(count):
            samples.append(self.generate_sample(i))
        return samples


if __name__ == "__main__":
    gen = SyntheticDatasetGenerator()
    samples = gen.generate_batch(10)
    print(f"Generated {len(samples)} samples in {gen.output_dir}")
