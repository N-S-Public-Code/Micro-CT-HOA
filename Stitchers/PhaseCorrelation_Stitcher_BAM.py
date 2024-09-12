import os
import cv2
import numpy as np
import tifffile

def phase_correlation(image1, image2):
    f1 = np.fft.fft2(image1)
    f2 = np.fft.fft2(image2)
    R = (f1 * f2.conjugate()) / np.abs(f1 * f2.conjugate())
    r = np.fft.ifft2(R)
    max_loc = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    shift_y, shift_x = max_loc
    if shift_y > image1.shape[0] // 2:
        shift_y -= image1.shape[0]
    if shift_x > image1.shape[1] // 2:
        shift_x -= image1.shape[1]
    return shift_x, shift_y
"""
def linear_blending(image1, image2, alpha=0.5):
    # Ensure both images have the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for blending")

    # Perform linear blending
    blended_img = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended_img
"""
def translate_image(image, shift_x, shift_y):
    rows, cols = image.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image

def stitch_images(image1, image2):
    shift_x, shift_y = phase_correlation(image1, image2)
    translated_image2 = translate_image(image2, shift_x, shift_y)
    # Determine the size of the stitched image
    h1, w1 = image1.shape
    h2, w2 = translated_image2.shape

    # Create a composite image to hold the result
    stitched_image = np.zeros((max(h1, h2), w1 + w2))

    # Place the first image in the composite image
    stitched_image[:h1, :w1] = image1
    
    #works but too high time complexity, improve it later
    # Blend the translated second image into the composite image 
    for y in range(h1):
        for x in range(w1, w1 + w2):
            if x - w1 < translated_image2.shape[1] and y < translated_image2.shape[0]:
                stitched_image[y, x] = max(stitched_image[y, x], translated_image2[y, x - w1])
    #return stitched_image

    # Corpping the image
    # Threshold the image to create a binary image
    _, binary_panorama = cv2.threshold(stitched_image, 1, 65535, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_panorama.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the panorama using the bounding box coordinates
    cropped_panorama = stitched_image[y:y+h, x:x+w].astype(np.uint16)
    # Corp to ROI
    x, y, w, h = ROI

    # Ensure the ROI is within the bounds of the image
    if x+w > cropped_panorama.shape[1] or y+h > cropped_panorama.shape[0]:
        print(f"Error: ROI {ROI} is out of bounds for image.")
    # Crop the image
    cropped_panorama = cropped_panorama[y:y+h, x:x+w]
    return cropped_panorama

def stitch_image_stacks (stack1, stack2):
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
file1 = r'E:\Users\nshubayev\Desktop\107_220422_1213_Disso_8p2_a_ref_Z40_Y4805_24500eV_0p72um_250ms\experiment\Substack1-1200.tif'
# Radiographs for 180-360 deg rotation (rigth side)
file2 = r'E:\Users\nshubayev\Desktop\107_220422_1213_Disso_8p2_a_ref_Z40_Y4805_24500eV_0p72um_250ms\experiment\sub1201-2400.tif'
saving_path = r"E:\Users\nshubayev\Desktop\107_220422_1213_Disso_8p2_a_ref_Z40_Y4805_24500eV_0p72um_250ms\experiment"
log_path = saving_path
stack1 = read_tiff_stack(file1)

#Rectangular Region of Interest
ROI = (70, 70, 2070, 1010) # Define your ROI (x, y, width, height)

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
print("the number of images in the first stack is: ")
print(len(stack1))
print("Reading the second stack...")
stack2 = read_tiff_stack(file2)
print("the number of images in the second stack is: ")
print(len(stack2))
print("stitching images using Phase Correlation...")

stitched_stack = stitch_image_stacks(stack1, stack2)
