"""
A program to split video frames in half and only save the right half
"""

# get all the image files in the directory
import os
from tqdm import tqdm
import cv2



directory_path = '/home/brownjordan317/fall_2024/CSCI443/Github/CSCI-443/object_detection/video_frames_30fps'
out_folder = 'right'
os.makedirs(out_folder, exist_ok=True)

# Get a list of image files in the directory
image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Loop over the image files
for image_file in tqdm(image_files):
    # Load the image
    image_path = os.path.join(directory_path, image_file)
    image = cv2.imread(image_path)

    # Split the image into two halves
    half_width = image.shape[1] // 2
    right_half = image[:, half_width:]

    # Save the right half
    output_path = os.path.join(out_folder, image_file)
    cv2.imwrite(output_path, right_half)