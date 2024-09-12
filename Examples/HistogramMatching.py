import os
from PIL import Image
import numpy as np
import cv2
import tifffile
import math
from skimage import exposure
from skimage import io
import scipy

def histogram_matching(source_image, reference_image):
    
    # Compute the histogram of the source and reference images
    source_hist, bins = np.histogram(source_image.flatten(), bins=65536, range=[0, 65536], density=True)
    reference_hist, bins = np.histogram(reference_image.flatten(), bins=65536, range=[0, 65536], density=True)
    # Compute the cumulative distribution function (CDF) for the source and reference images
    source_cdf = np.cumsum(source_hist)
    reference_cdf = np.cumsum(reference_hist)
    #source_cdf = scipy.stats.norm.cdf(source_hist)
    #reference_cdf = scipy.stats.norm.cdf(reference_hist)
    # Create a lookup table to map pixel values from source to reference
    lookup_table = np.zeros(65536, dtype=np.uint16)
    reference_cdf_index = 0
    
    for source_cdf_index in range(65536):
        while reference_cdf_index < 65536 and reference_cdf[reference_cdf_index] < source_cdf[source_cdf_index]:
            reference_cdf_index += 1
        lookup_table[source_cdf_index] = reference_cdf_index

    # Apply the lookup table to the source image to get the matched image
    matched = lookup_table[source_image]
    
    return matched
    
    """
    cdf_source = compute_cdf(source_image)
    cdf_reference = compute_cdf(reference_image)

    # Create a lookup table to map the pixel values
    lookup_table = np.interp(cdf_source, cdf_reference, np.arange(65536))

    # Apply the lookup table to the source image
    matched = np.interp(source_image.flatten(), np.arange(65536), lookup_table).reshape(source_image.shape)
    matched = matched.astype(np.uint16)
    return matched
    """
    """
    matched = exposure.match_histograms(source_image, reference_image, multichannel=True)
    return matched.astype(np.int16)
    """
    

def compute_cdf(image):
        hist, _ = np.histogram(image.flatten(), 65536, [0, 65536])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 65535 / cdf[-1]  # Normalize to 0-65535
        return cdf_normalized

def read_tiff_images(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter out relevant TIFF files
    tiff_files = [f for f in files if f.endswith('.tiff') or f.endswith('.tif')]
    # Read and process each TIFF file
    projArr = []
    for file in tiff_files:
        file_path = os.path.join(folder_path, file)
        img = io.imread(file_path)
        #img = tifffile.imread(file_path)
        projArr.append(img)
        
        # Display image information
        print(f'Read {file}')
    #numpy_arrays = [np.array(image) for image in projArr]
    return projArr

def count_proj_in_folder(folder_path):
    try:
        # List all entries in the given folder
        entries = os.listdir(folder_path)
        
        # Filter out directories and count files
        file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(folder_path, entry)))
        
        return file_count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def create_subfolder(path, subfolder_name):
    # Construct the full path for the subfolder
    subfolder_path = os.path.join(path, subfolder_name)
    
    try:
        # Create the subfolder
        os.makedirs(subfolder_path, exist_ok=True)
        print(f"Subfolder '{subfolder_name}' created at '{path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

def HM_CorespondingsAndSave(projArr):
    print("Calculating and matching histograms...")
    matchedArr = []
    for i in range(math.floor(number_of_proj/2)):
        matchedArr.append(histogram_matching(projArr[i],projArr[i+math.floor(number_of_proj/2)]))
    return matchedArr
"""
Usage
"""
# Path to working folder, all the projections should start with 'proj'
folder_path = r"E:\Users\nshubayev\Desktop\GHM4Recon"
number_of_proj = count_proj_in_folder(folder_path)
tiff_images = read_tiff_images(folder_path)
matchedHisto = HM_CorespondingsAndSave(tiff_images)
# Saving folder
create_subfolder(folder_path, 'HM_proj')

num_images = len(matchedHisto)
for i in range(num_images):
    # Save stitched images as TIFF or process further as needed
    file_name = "image_{:04d}.tif".format(i)
    tifffile.imwrite(os.path.join(os.path.join(folder_path, 'HM_proj'),file_name), matchedHisto[i])
    print(f'HM image {i+1} saved successfully.')
print("Done")