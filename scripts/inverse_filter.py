import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt

# Generate Motion Blur PSF
def generate_motion_psf(length, angle, size):
    """Generate a motion blur Point Spread Function (PSF)."""
    psf = np.zeros((size, size))
    center = size // 2
    angle_rad = np.deg2rad(angle)
    for i in range(length):
        x = center + int(i * np.cos(angle_rad))
        y = center + int(i * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1
    psf /= psf.sum()  # Normalize
    return psf

# Apply Motion Blur
def apply_motion_blur(image, psf):
    """Apply motion blur to each channel of a color image."""
    blurred_channels = []
    for channel in cv2.split(image):
        blurred = convolve2d(channel, psf, mode="same", boundary="wrap")
        blurred_channels.append(blurred)
    blurred_image = cv2.merge(blurred_channels)
    return blurred_image

# Inverse Filter Restoration
def inverse_filter(degraded, psf, eps=1e-3):
    """Restore an image using the inverse filter."""
    psf_fft = fft2(psf, s=degraded.shape[:2])
    psf_fft[np.abs(psf_fft) < eps] = eps  # Avoid division by zero
    restored_channels = []
    for channel in cv2.split(degraded):
        degraded_fft = fft2(channel)
        restored_fft = degraded_fft / psf_fft
        restored = np.real(ifft2(restored_fft))
        restored_channels.append(restored)
    restored_image = cv2.merge(restored_channels)
    return restored_image

# Main Code
if __name__ == "__main__":
    # Load the original color image
    original = cv2.imread(r"C:\Users\sujaan iqbal\OneDrive\Desktop\grpcv\zeproject\clean_images\catty2.jpg")
    if original is None:
        raise FileNotFoundError("Image not found. Check the file path.")

    # Normalize the image to [0, 1]
    original_normalized = original.astype(np.float32) / 255.0

    # Generate the motion PSF
    psf = generate_motion_psf(length=15, angle=30, size=25)  # Adjust length, angle, and size as needed

    # Apply motion blur
    blurred = apply_motion_blur(original_normalized, psf)

    # Restore the image using the inverse filter
    restored = inverse_filter(blurred, psf)

    # Display the images using matplotlib
    images = [original, (blurred * 255).astype(np.uint8), (restored * 255).astype(np.uint8)]
    titles = ["Original Image", "Motion Blurred Image", "Restored Image (Inverse Filter)"]

    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i + 1)
        if i == 0:  # Original image is in BGR
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:  # Processed images are already in RGB
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
