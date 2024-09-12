import skimage
from skimage import io, measure, color
import matplotlib.pyplot as plt

# Load two images, assuming they are tomogtams (grayscale)
im1 = r"E:\Users\path\to\your\image1.tiff"
im2 = r"E:\Users\path\to\your\image2.tiff"
image1 = io.imread(im1, as_gray=True)
image2 = io.imread(im2, as_gray=True)

# Measure blur effect using the skimage measure.blur_effect
blur_image1 = skimage.measure.blur_effect(image1)
blur_image2 = skimage.measure.blur_effect(image2)

# Display the results
print(f"Blur effect of Image 1: {blur_image1}")
print(f"Blur effect of Image 2: {blur_image2}")

# Optionally, display the images for visual comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image1, cmap='gray')
ax[0].set_title(f"Image 1 - Blur: {blur_image1:.2f}")

ax[1].imshow(image2, cmap='gray')
ax[1].set_title(f"Image 2 - Blur: {blur_image2:.2f}")

plt.show()
