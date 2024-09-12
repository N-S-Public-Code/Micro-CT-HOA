import cv2
import numpy as np
from matplotlib import pyplot as plt
import tifffile

#Function to save a 16-bit grayscale TIFF image
def save_tiff_image(image, filepath):
    tifffile.imwrite(filepath, image.astype(np.uint16))

# Function to perform log-polar phase correlation and return rotation and scale
def log_polar_phase_correlation(img1, img2):
    # FFT of both images
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    
    # Magnitude spectra of both images
    mag1 = np.abs(f1)
    mag2 = np.abs(f2)

    # Convert magnitude spectra to log-polar coordinates
    center1 = (mag1.shape[1]//2, mag1.shape[0]//2)
    center2 = (mag2.shape[1]//2, mag2.shape[0]//2)
    
    log_polar1 = cv2.logPolar(mag1, center1, 40, cv2.WARP_FILL_OUTLIERS)
    log_polar2 = cv2.logPolar(mag2, center2, 40, cv2.WARP_FILL_OUTLIERS)
    
    # Phase correlation in log-polar domain to estimate rotation and scaling
    cross_power_spectrum = (log_polar1 * np.conj(log_polar2)) / np.abs(log_polar1 * np.conj(log_polar2))
    phase_corr = np.fft.ifft2(cross_power_spectrum)
    max_idx = np.unravel_index(np.argmax(np.abs(phase_corr)), phase_corr.shape)
    
    # Rotation and scale estimation
    rotation = max_idx[1] * 360 / log_polar1.shape[1]  # Convert index to angle
    scale = np.exp(max_idx[0] / log_polar1.shape[0])  # Convert log distance to scale
    
    return rotation, scale

# Function to compute translation using phase correlation
def compute_translation(img1, img2):
    # FFT of both images
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)

    # Compute cross power spectrum
    cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
    translation_corr = np.fft.ifft2(cross_power_spectrum)
    max_idx = np.unravel_index(np.argmax(np.abs(translation_corr)), translation_corr.shape)

    # Translation estimation (x, y)
    translation_x = max_idx[1] - img1.shape[1] if max_idx[1] > img1.shape[1] // 2 else max_idx[1]
    translation_y = max_idx[0] - img1.shape[0] if max_idx[0] > img1.shape[0] // 2 else max_idx[0]

    return (translation_x, translation_y)

# Main function to stitch images
def stitch_images(image1, image2):
    # Step 1: Estimate rotation and scaling using log-polar phase correlation
    rotation, scale = log_polar_phase_correlation(image1, image2)
    print ("rottion: ")
    print (rotation)
    print ("scale: ")
    print (scale)
    # Step 2: Correct scaling and rotation in image2
    center = (image2.shape[1] // 2, image2.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, -rotation, 1 / scale)
    aligned_image2 = cv2.warpAffine(image2, M, (image2.shape[1], image2.shape[0]))

    
    # Step 3: Compute translation between image1 and aligned_image2
    translation_x, translation_y = compute_translation(image1, aligned_image2)
    print ("delta X: ")
    print (translation_x)
    print ("delta Y: ")
    print (translation_y)
    
    # Step 4: Create a new canvas to hold the stitched images
    stitched_height = max(image1.shape[0], aligned_image2.shape[0] + abs(translation_y))
    stitched_width = max(image1.shape[1], aligned_image2.shape[1] + abs(translation_x))
    stitched_image = np.zeros((stitched_height, stitched_width), dtype=np.uint16)

    # Step 5: Place image1 in the stitched image
    stitched_image[:image1.shape[0], :image1.shape[1]] = image1

    # Step 6: Place aligned_image2 into the stitched image with the correct translation
    y_start = max(abs(translation_y), 0)
    x_start = max(abs(translation_x), 0)
    y_end = min(y_start + aligned_image2.shape[0], stitched_image.shape[0])
    x_end = min(x_start + aligned_image2.shape[1], stitched_image.shape[1])

    aligned_image2_cropped = aligned_image2[:y_end - y_start, :x_end - x_start]
    stitched_image[y_start:y_end, x_start:x_end] = aligned_image2_cropped
    
    return stitched_image
def stitch_images_with_blending(image1, image2):
    # Step 1: Estimate rotation and scaling using log-polar phase correlation
    rotation, scale = log_polar_phase_correlation(image1, image2)

    # Step 2: Correct scaling and rotation in image2
    center = (image2.shape[1] // 2, image2.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, -rotation, 1 / scale)
    aligned_image2 = cv2.warpAffine(image2, M, (image2.shape[1], image2.shape[0]))

    # Step 3: Compute translation between image1 and aligned_image2
    translation_x, translation_y = compute_translation(image1, aligned_image2)

    # Step 4: Create a new canvas to hold the stitched images
    stitched_height = max(image1.shape[0], aligned_image2.shape[0] + abs(translation_y))
    stitched_width = max(image1.shape[1], aligned_image2.shape[1] + abs(translation_x))
    stitched_image = np.zeros((stitched_height, stitched_width), dtype=np.uint16)

    # Step 5: Place image1 in the stitched image
    stitched_image[:image1.shape[0], :image1.shape[1]] = image1

    # Step 6: Determine the overlapping region and perform blending
    y_start = max(abs(translation_y), 0)
    x_start = max(translation_x, 0)
    y_end = min(y_start + aligned_image2.shape[0], stitched_image.shape[0])
    x_end = min(x_start + aligned_image2.shape[1], stitched_image.shape[1])

    # Crop aligned_image2 to fit within the stitched image
    aligned_image2_cropped = aligned_image2[:y_end - y_start, :x_end - x_start]

    # Define the overlapping region
    #overlap_x_start = max(image1.shape[1], x_start)
    #overlap_x_end = min(x_end, image1.shape[1])
    overlap_width = abs(x_end) - abs(x_start)
    overlap_width = abs(translation_x)

    # Place the non-overlapping part of aligned_image2
    stitched_image[y_start:y_end, x_start:x_end] = aligned_image2_cropped

    # Perform linear blending in the overlapping region
    blended = cv2.addWeighted(stitched_image, 0.5, aligned_image2_cropped, 1 - 0.5, 0)
    """
    for i in range(overlap_width):
        alpha = i / overlap_width  # Linear weight factor
        blended_value =  alpha* stitched_image[y_start:y_end, x_start + i] + \
                    (1 - alpha)* aligned_image2_cropped[:, i]
        stitched_image[y_start:y_end, x_start + i] = blended_value
        return stitched_image
    """
    return blended

    

# Load the two 16-bit TIFF grayscale images
im1_path = r"F:\Natanel_DO_NOT_DELETE\ESRF_Binned_SuperCleanRecon\68\HOA\Stitched_HM_LogPolar_PhaseCorr\left.tif"
im2_path = r"F:\Natanel_DO_NOT_DELETE\ESRF_Binned_SuperCleanRecon\68\HOA\Stitched_HM_LogPolar_PhaseCorr\rigth.tif"

image1 = cv2.imread(im1_path, cv2.IMREAD_UNCHANGED)
image2 = cv2.imread(im2_path, cv2.IMREAD_UNCHANGED)

# Ensure the images are 16-bit grayscale
if image1 is None or image2 is None:
    print("Error loading images. Ensure the paths are correct and images are 16-bit TIFF.")
    exit()

if image1.dtype != np.uint16 or image2.dtype != np.uint16:
    print("Input images are not 16-bit grayscale.")
    exit()

# Perform stitching
stitched_image = stitch_images_with_blending(image1, image2)

# Save the result as a 16-bit TIFF
save_tiff_image(stitched_image, 'stitched_image1.tif')
# Optionally, display the stitched image (for visualization purposes)
plt.imshow(stitched_image, cmap='gray')
plt.title("Stitched Image")
plt.show()
