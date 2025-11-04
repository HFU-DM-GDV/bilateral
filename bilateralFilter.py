# bilateralFilter.py
"""
Demonstration of Bilateral Filter Kernel Computation and Visualization. 
It provides an interactive plot where clicking on an intensity patch
updates the displayed bilateral filter kernel based on the selected 
pixel's intensity. It is inspired by the illustration from 
C. Tomasi and R. Manduchi, "Bilateral filtering for gray and color images," 
Sixth International Conference on Computer Vision (IEEE Cat. No.98CH36271), 
Bombay, India, 1998, pp. 839-846, doi: 10.1109/ICCV.1998.710815.,
https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf
Vibe coded by: Uwe Hahne
Date: November 2025
"""

import numpy as np
import cv2

import matplotlib.pyplot as plt

def spatial_gaussian(x, y, sigma_s):
    return np.exp(-(x**2 + y**2) / (2 * sigma_s**2))

def range_gaussian(img, img_center, sigma_r):
    return np.exp(-((img - img_center) ** 2) / (2 * sigma_r**2))

def make_step_patch(n, step_col=None, low=0.0, high=1.0):
    if step_col is None:
        step_col = n // 2
    I = np.full((n, n), low, dtype=float)
    I[:, step_col:] = high
    return I

def compute_bilateral_kernel(image, n_kernel=51, sigma_s=7.0, sigma_r=0.1):
    n = image.shape[0]
    cy = cx = n // 2
    y, x = np.mgrid[-n_kernel//2:n_kernel//2, -n_kernel//2:n_kernel//2]

    gauss_spatial = spatial_gaussian(x, y, sigma_s)
    I0 = image[cy, cx]
    img_patch = image[cy - n_kernel//2:cy + n_kernel//2 + 1, cx - n_kernel//2:cx + n_kernel//2 + 1]
    gauss_range = range_gaussian(img_patch, I0, sigma_r)

    K = gauss_spatial * gauss_range
    K /= K.sum() + 1e-12
    return x, y, K, gauss_spatial, gauss_range

def main():
    # Parameters
    n_kernel = 31   # kernel size (odd)
    n = n_kernel * 10  # image size (10x kernel)
    sigma_s = 7.0   # spatial sigma (pixels)
    sigma_r = 0.10  # range sigma (intensity units, assuming [0,1])

    # Synthetic intensity patch: vertical step edge
    img_path = 'images/Bumbu_Rawon.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Check if image is loaded
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    img = cv2.resize(img, (n, n), interpolation=cv2.INTER_LANCZOS4)
    I = np.asarray(img, dtype=float) / 255.0

    X, Y, K, gauss_spatial, _ = compute_bilateral_kernel(I, n_kernel=n_kernel, sigma_s=sigma_s, sigma_r=sigma_r)

    # Plot: intensity patch and 3D bilateral kernel
    fig = plt.figure(figsize=(12, 5))

    ax_img = fig.add_subplot(1, 2, 1)
    _ = ax_img.imshow(I, cmap='gray', vmin=0, vmax=1, origin='upper')
    ax_img.set_title('Intensity patch (step edge)')
    def _on_click(event):
        if event.inaxes is not ax_img or event.xdata is None or event.ydata is None:
            return
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        i = max(0, min(I.shape[0] - 1, i))
        j = max(0, min(I.shape[1] - 1, j))

        I0 = I[i, j]
        img_patch = I[i - n_kernel//2:i + n_kernel//2 + 1, j - n_kernel//2:j + n_kernel//2 + 1]
        gauss_range = range_gaussian(img_patch, I0, sigma_r)
        bilateral_kernel = gauss_spatial * gauss_range
        bilateral_kernel /= bilateral_kernel.sum() + 1e-12

        for coll in ax_img.collections:
            if coll.get_label() == 'center':
                coll.set_offsets([[j, i]]) # pyright: ignore[reportAttributeAccessIssue]
                break

        ax3d.clear()
        ax3d.plot_surface(X, Y, bilateral_kernel, cmap='viridis', linewidth=0, antialiased=True) # pyright: ignore[reportAttributeAccessIssue]
        ax3d.set_title('Bilateral filter kernel (normalized)')
        ax3d.set_xlabel('x offset')
        ax3d.set_ylabel('y offset')
        ax3d.set_zlabel('weight') # pyright: ignore[reportAttributeAccessIssue]
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    ax_img.scatter([I.shape[1]//2], [I.shape[0]//2], c='r', s=30, label='center')
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.legend(loc='lower right', fontsize=8)

    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax3d.plot_surface(X, Y, K, cmap='viridis', linewidth=0, antialiased=True) # pyright: ignore[reportAttributeAccessIssue]
    ax3d.set_title('Bilateral filter kernel (normalized)')
    ax3d.set_xlabel('x offset')
    ax3d.set_ylabel('y offset')
    ax3d.set_zlabel('weight') # pyright: ignore[reportAttributeAccessIssue]
    fig.colorbar(surf, ax=ax3d, shrink=0.6, pad=0.1)

    fig.suptitle(f'Bilateral Kernel: sigma_s={sigma_s}, sigma_r={sigma_r}', fontsize=12)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()