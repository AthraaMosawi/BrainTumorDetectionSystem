import numpy as np
import cv2

class MWIReconstruction:
    """
    Implements the Delay-and-Sum (DAS) algorithm for Microwave Imaging.
    Converts scattered signals back into a dielectric map.
    """
    def __init__(self, c=3e8, dx=0.005):
        self.c = c
        self.dx = dx
        
    def delay_and_sum(self, signals, antenna_positions, grid_size=(128, 128)):
        """
        signals: shape (num_antennas, time_steps)
        antenna_positions: list of (x, y)
        """
        reconstructed = np.zeros(grid_size)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                pixel_pos = np.array([i * self.dx, j * self.dx])
                sum_val = 0
                for ant_idx, ant_pos in enumerate(antenna_positions):
                    dist = np.linalg.norm(pixel_pos - ant_pos)
                    delay = dist / self.c
                    # Simplified: in a real case, we'd index into the time-domain signal
                    # based on the delay.
                    sum_val += signals[ant_idx][int(delay * 1e11) % len(signals[ant_idx])]
                reconstructed[i, j] = sum_val**2
        
        # Normalize
        reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)
        return reconstructed.astype(np.uint8)

class ImageProcessor:
    @staticmethod
    def enhance(image):
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        return enhanced

    @staticmethod
    def register_images(moving, fixed):
        """
        Simplified registration using ORB features and homography.
        In a real med-imaging app, MI (Mutual Information) would be used.
        """
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(moving, None)
        kp2, des2 = orb.detectAndCompute(fixed, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        registered = cv2.warpPerspective(moving, M, (fixed.shape[1], fixed.shape[0]))
        return registered
