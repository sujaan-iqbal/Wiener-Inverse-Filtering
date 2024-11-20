import numpy as np
import matplotlib.pyplot as plt



def generate_psf(size, std):
    """Generate a Gaussian Point Spread Function (PSF)."""
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    psf = np.exp(-(x**2 + y**2) / (2 * std**2))
    psf /= psf.sum()  # Normalize
    return psf

def save_psf_as_csv(psf, file_path):
    """Save the PSF matrix to a CSV file."""
    np.savetxt(file_path, psf, delimiter=",")
    print(f"PSF saved to {file_path}")

# Generate and save the PSF
psf = generate_psf(size=5, std=20)  # Example PSF
save_psf_as_csv(psf, "psfs/psf_average.csv")

plt.imshow(psf, cmap='gray')
plt.title('Point Spread Function (PSF)')
plt.colorbar()
plt.show()
