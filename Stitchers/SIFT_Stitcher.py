import os
import cv2
import numpy as np
import tifffile

def stitch_images(image1, image2):
    
    # Create SIFT detector and detect keypoints and descriptors
    sift = cv2.SIFT_create()
    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute((image1/256).astype('uint8'), None)
    keypoints2, descriptors2 = sift.detectAndCompute((image2/256).astype('uint8'), None)

    # Use BFMatcher to find the best matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Find the homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the second image to align with the first
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    panorama_width = width1 + width2
    panorama_height = max(height1, height2)
    _, invertedH = cv2.invert(H)
    #dst = cv2.perspectiveTransform(image2, invertedH)
    panorama = cv2.warpPerspective(image2, invertedH, (panorama_width, panorama_height))
    panorama[0:height1, 0:width1] = image1

    # Corpping the image
    # Threshold the image to create a binary image
    _, binary_panorama = cv2.threshold(panorama, 1, 65535, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_panorama.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the panorama using the bounding box coordinates
    cropped_panorama = panorama[y:y+h, x:x+w]

    # Corp to ROI
    x, y, w, h = ROI

    # Ensure the ROI is within the bounds of the image
    if x+w > cropped_panorama.shape[1] or y+h > cropped_panorama.shape[0]:
        print(f"Error: ROI {ROI} is out of bounds for image.")
    # Crop the image
    cropped_panorama = cropped_panorama[y:y+h, x:x+w] 
    return cropped_panorama

def stitch_image_stacks(stack1, stack2):
    """Stitches two stacks of images pairwise."""
    if len(stack1) != len(stack2):
        print("Error: Number of images in the two stacks do not match.")
        return None
    num_images = len(stack1)
    for i in range(num_images):
        stitched_image = stitch_images(stack1[i], stack2[i])
        # Save stitched images as TIFF or process further as needed
        file_name = "image_{:04d}.tif".format(i)
        tifffile.imwrite(os.path.join(saving_path,file_name), stitched_image)
        print(f'Stitched image {i+1} saved successfully.')
    #Generate log file for further reconstruction via Nrecon or documenting 
    generateLogFile(os.path.join(saving_path,file_name), log_path)

def read_tiff_stack(filename):
    """Reads a stack of images from a TIFF file."""
    img = tifffile.imread(filename)
    return img
def generateLogFile(tiff_path, log_path):
    file_name = "image_.log"
    logFile = os.path.join(log_path, file_name)
    # Check if the TIFF file exists
    if not os.path.exists(tiff_path):
        raise Exception(f'TIFF file not found: {tiff_path}')

    # Open the log file for writing
    with open(logFile, 'w') as f:
        # Write the reconstruction parameters to the log file
        f.write(f'[System]\n')
        f.write(f'Scanner= {Scanner}\n')
        f.write(f'[Acquisition]\n')
        f.write(f'Optical Axis (line)= {Optical_Axis}\n')
        f.write(f'Object to Source (mm)= {Object_to_Source}\n')
        #number of files == number of images:
        f.write(f'Number of files= {Number_of_files}\n')
        f.write(f'Image Pixel Size (um)= {Image_Pixel_Size}\n')
        f.write(f'Linear pixel value= {Linear_pixel_value}\n')
        f.write(f'Rotation Step (deg)= {Rotation_Step}\n')
        f.write(f'Use 360 Rotation= {Use_360_Rotation}\n')
        f.write(f'Rotation Direction= {Rotation_Direction}\n')
        f.write(f'Flat Field Correction= {Flat_Field_Correction}\n')
        f.write(f'FF updating interval= {FF_updating_interval}\n')
        f.write(f'[Reconstruction]\n')
        f.write(f'Pseudo-parallel projection calculated= {Pseudo_parallel_projection_calculated}\n')
    print("log file saved successfully.")

# Usage

# Radiographs for 0-180 deg rotation (left side)
file1 = r'F:\Natanel_DO_NOT_DELETE\ESRF_Binned_SuperClean\69\Stitched(HM_SIFT)\stack1.tif'
# Radiographs for 180-360 deg rotation (rigth side)
file2 = r'F:\Natanel_DO_NOT_DELETE\ESRF_Binned_SuperClean\69\Stitched(HM_SIFT)\stack2.tif'
saving_path = r"F:\Natanel_DO_NOT_DELETE\ESRF_Binned_SuperClean\69\Stitched(HM_SIFT)"
log_path = saving_path
stack1 = read_tiff_stack(file1)

# Parameters
MIN_MATCH_COUNT = 8  # Adjust this threshold as needed
ratio = 0.75
#Rectangular Region of Interest
ROI = (60, 60, 2050, 960) # Define your ROI (x, y, width, height)

# Set the reconstruction parameters according to Pauls instruction
Scanner = 'ESRF_ID19_Beamline'
Optical_Axis = 1000
Object_to_Source = 10000000
Number_of_files = len(stack1)
Image_Pixel_Size = 'pixelSize'
Linear_pixel_value = 0
Rotation_Step = 180/Number_of_files
Use_360_Rotation = 'NO'
Rotation_Direction = 'CC'
Flat_Field_Correction = 'ON'
FF_updating_interval = 500
Pseudo_parallel_projection_calculated= 1

# Masseges to the user
print("Reading the first stack...")
stack1 = read_tiff_stack(file1)
print("the number of images in the first stack is: ")
print(len(stack1))
print("Reading the second stack...")
stack2 = read_tiff_stack(file2)
print("the number of images in the second stack is: ")
print(len(stack2))

stitched_stack = stitch_image_stacks(stack1, stack2)

