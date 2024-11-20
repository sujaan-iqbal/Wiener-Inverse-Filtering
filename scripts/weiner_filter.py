import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2

# Step 1: Generate Motion Blur PSF
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
    psf /= psf.sum()
    return psf

# Step 2: Simulate Motion Blur
def apply_motion_blur(image, psf):
    """Simulate motion blur by convolving the image with the PSF."""
    blurred = convolve2d(image, psf, mode="same", boundary="wrap")
    return blurred

# Step 3: Add Gaussian Noise
def add_gaussian_noise(image, mean=0, var=0.001):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

# Step 4: Wiener Filter
def wiener_filter(degraded, psf, nsr):
    """Perform Wiener deconvolution."""
    psf_fft = fft2(psf, s=degraded.shape)
    degraded_fft = fft2(degraded)
    psf_fft_conj = np.conj(psf_fft)

    # Wiener filter formula
    restored_fft = (psf_fft_conj / (np.abs(psf_fft) ** 2 + nsr)) * degraded_fft
    restored = np.real(ifft2(restored_fft))
    return np.clip(restored, 0, 1)

# Step 5: Main Function to Test
if __name__ == "__main__":
    # Load the original image (grayscale and normalized)
    original = cv2.imread(r"C:\Users\sujaan iqbal\OneDrive\Desktop\grpcv\zeproject\clean_images\gwirly.jpg", cv2.IMREAD_GRAYSCALE) / 255.0

    # Generate a motion PSF
    psf = generate_motion_psf(length=21, angle=11, size=25)

    # Apply motion blur
    blurred = apply_motion_blur(original, psf)

    # Add Gaussian noise
    blurred_noisy = add_gaussian_noise(blurred, mean=0, var=0.01)

    # Apply Wiener filter (estimate NSR based on noise variance)
    signal_var = np.var(original)
    noise_var = 0.0001
    nsr = noise_var / signal_var
    restored = wiener_filter(blurred_noisy, psf, nsr)

    # Save results
    cv2.imwrite("blurred_image.png", (blurred * 255).astype(np.uint8))
    cv2.imwrite("blurred_noisy_image.png", (blurred_noisy * 255).astype(np.uint8))
    cv2.imwrite("restored_image.png", (restored * 255).astype(np.uint8))

    # Display results
    cv2.imshow("Original", original)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Blurred + Noisy", blurred_noisy)
    cv2.imshow("Restored", restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
