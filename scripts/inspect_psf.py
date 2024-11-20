import numpy as np
import os

# Define the path to the PSF file
psf_path = os.path.join("..", "psfs", "psf_average.csv")  # Adjust as needed

try:
    # Check if the file exists
    if not os.path.exists(psf_path):
        raise FileNotFoundError(f"PSF file not found at path: {psf_path}")

    # Load the file with allow_pickle=True to inspect content
    psf = np.load(psf_path, allow_pickle=True)
    print(f"File type: {type(psf)}")
    print(f"File content: {psf}")
except Exception as e:
    print(f"Error loading file: {e}")
