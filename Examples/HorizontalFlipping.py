import os
from PIL import Image
import numpy as np
import cv2
import tifffile
import math

def read_tiff_images(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter out relevant TIFF files (#proj/2)
    tiff_files = [f for f in files if f.endswith('.tiff') or f.endswith('.tif')]
    # Half of the total num of projections
    #tiff_files = tiff_files[:(math.floor(number_of_proj/2))]
    # Read and process each TIFF file
    projArr = []
    for file in tiff_files:
        file_path = os.path.join(folder_path, file)
        img = tifffile.imread(file_path)
        projArr.append(img)
        
        # Display image information
        print(f'Read {file}')
    #numpy_arrays = [np.array(image) for image in projArr]
    return projArr

def horizontal_flipper(images):
    # Flip the image horizontally
    flipped_image = [cv2.flip(image, 1) for image in images]
    return flipped_image

def create_subfolder(path, subfolder_name):
    # Construct the full path for the subfolder
    subfolder_path = os.path.join(path, subfolder_name)
    
    try:
        # Create the subfolder
        os.makedirs(subfolder_path, exist_ok=True)
        print(f"Subfolder '{subfolder_name}' created at '{path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

def count_proj_in_folder(folder_path):
    try:
        # List all entries in the given folder
        entries = os.listdir(folder_path)
        
        # Filter out directories and count files
        file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(folder_path, entry)) and entry.startswith('DFFC'))
        
        return file_count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
# Usage 
# Path to working folder, all the projections should start with 'proj'
folder_path = r"F:\Natanel_DO_NOT_DELETE\ESRF_Binned_SuperClean\68\DFFC\HM_proj"
number_of_proj = count_proj_in_folder(folder_path)
tiff_images = read_tiff_images(folder_path)
flipped = horizontal_flipper(tiff_images)
# Saving folder
create_subfolder(folder_path, 'flipped_proj')

num_images = len(flipped)
for i in range(num_images):
    # Save stitched images as TIFF or process further as needed
    file_name = "image_{:04d}.tif".format(i)
    tifffile.imwrite(os.path.join(os.path.join(folder_path, 'flipped_proj'),file_name), flipped[i])
    print(f'Flipped image {i+1} saved successfully.')
