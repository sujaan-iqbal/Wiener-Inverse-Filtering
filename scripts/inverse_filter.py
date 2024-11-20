import numpy as np
import cv2
import os


def pad_psf(psf, shape):
    """Pad the PSF to match the image shape."""
    padded_psf = np.zeros(shape)
    padded_psf[:psf.shape[0], :psf.shape[1]] = psf
    return padded_psf

def restore_images(degraded_folder, restored_folder, psf_csv_path):
    """Apply inverse filter to all images in the degraded folder."""
    # Load the PSF from CSV
    psf = np.loadtxt(psf_csv_path, delimiter=",")
    
    if not os.path.exists(restored_folder):
        os.makedirs(restored_folder)

    for filename in os.listdir(degraded_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # Load degraded image
            img_path = os.path.join(degraded_folder, filename)
            degraded_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # FFT of the degraded image
            image_fft = np.fft.fft2(degraded_img)

            # Pad the PSF to match image dimensions
            psf_padded = pad_psf(psf, degraded_img.shape)

            # FFT of the PSF
            psf_fft = np.fft.fft2(psf_padded)

            # Avoid division by zero and apply inverse filter
            restored_fft = image_fft / (psf_fft + 1e-8)
            
            # Inverse FFT to get the restored image
            restored_img = np.abs(np.fft.ifft2(restored_fft))

            # Rescale the image to 0-255
            restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

            # Save the restored image
            output_path = os.path.join(restored_folder, filename)
            cv2.imwrite(output_path, restored_img)
            print(f"Restored image saved to {output_path}")

if __name__ == "__main__":
    restore_images("degraded_images", "restored_images", "psfs/psf_average.csv")
