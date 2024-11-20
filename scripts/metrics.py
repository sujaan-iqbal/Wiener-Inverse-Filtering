import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(original_img, restored_img):
    """Calculate PSNR and SSIM metrics."""
    psnr_value = psnr(original_img, restored_img, data_range=255)
    ssim_value, _ = ssim(original_img, restored_img, data_range=255, full=True)
    return psnr_value, ssim_value

if __name__ == "__main__":
    # Example usage
    original = cv2.imread(r"C:\Users\sujaan iqbal\OneDrive\Desktop\grpcv\zeproject\clean_images\catty2.jpg", cv2.IMREAD_GRAYSCALE)
    restored = cv2.imread(r"C:\Users\sujaan iqbal\OneDrive\Desktop\grpcv\zeproject\restored_images\catty2.jpg", cv2.IMREAD_GRAYSCALE)
    
    psnr_val, ssim_val = calculate_metrics(original, restored)
    print(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")