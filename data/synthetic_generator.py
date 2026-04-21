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

    def generate_sample(self, sample_id, class_label=None):
        """
        class_label: 0 = Normal, 1 = Cancer, 2 = Malformed
        """
        if class_label is None:
            class_label = random.choice([0, 1, 2])
            
        mri = np.zeros((128, 128), dtype=np.uint8)
        xray = np.zeros((128, 128), dtype=np.uint8)
        mwi = np.zeros((128, 128), dtype=np.uint8)

        # Baseline background
        mri.fill(random.randint(5, 15))
        xray.fill(random.randint(2, 8))
        mwi.fill(random.randint(2, 10))

        if class_label == 0:  # Normal
            # Standard brain
            cv2.circle(mri, (64, 64), 50, 40, -1)
            cv2.rectangle(xray, (20, 20), (108, 108), 80, 2)
            cv2.circle(mwi, (64, 64), 50, 30, -1)
            
        elif class_label == 1:  # Cancer
            # Standard brain with bright mass
            tumor_pos = (random.randint(35, 93), random.randint(35, 93))
            tumor_radius = random.randint(7, 14)
            
            cv2.circle(mri, (64, 64), 50, 40, -1)
            cv2.circle(mri, tumor_pos, tumor_radius, 240, -1)
            
            cv2.rectangle(xray, (20, 20), (108, 108), 80, 2)
            cv2.circle(xray, tumor_pos, tumor_radius, 200, -1)
            
            cv2.circle(mwi, (64, 64), 50, 30, -1)
            cv2.circle(mwi, tumor_pos, tumor_radius + 3, 230, -1)
            
        elif class_label == 2:  # Malformed
            # Structurally asymmetrical / shifted phantom
            shift_x = random.randint(-15, 15)
            shift_y = random.randint(10, 25)
            
            # Asymmetric ellipses and distorted skull
            cv2.ellipse(mri, (64 + shift_x, 64 + shift_y), (50, 30), random.randint(0, 45), 0, 360, 40, -1)
            cv2.rectangle(xray, (20 + shift_x, 20), (108, 108 + shift_y), 80, 2)
            cv2.ellipse(mwi, (64 + shift_x, 64 + shift_y), (50, 30), random.randint(0, 45), 0, 360, 30, -1)

        mri = cv2.GaussianBlur(mri, (3, 3), 0)
        xray = cv2.GaussianBlur(xray, (3, 3), 0)
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
            'class_label': class_label,
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
