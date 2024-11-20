import cv2
import numpy as np
import os

def add_blur_and_noise(image, psf_size=(5, 5), noise_std=10):
    # Add blur using a uniform Point Spread Function (PSF)
    psf = np.ones(psf_size) / (psf_size[0] * psf_size[1])
    blurred = cv2.filter2D(image, -1, psf)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, blurred.shape).astype(np.float32)
    degraded = blurred + noise
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)

    return degraded

def degrade_images(input_folder, output_folder, psf_size=(5, 5), noise_std=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        degraded = add_blur_and_noise(image, psf_size, noise_std)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, degraded)

# Example usage:
if __name__ == "__main__":
    degrade_images("clean_images", "degraded_images", psf_size=(5, 5), noise_std=20)
