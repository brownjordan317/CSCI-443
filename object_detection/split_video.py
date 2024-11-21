import subprocess
import os

def video_to_images(video_path, output_folder, image_format, fps):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the ffmpeg command
    command = [
        'ffmpeg', 
        '-i', video_path,             # Input video path
        '-vf', f'fps={fps}',          # Frame rate for image extraction
        os.path.join(output_folder, f'frame_%04d.{image_format}')  # Output file pattern
    ]
    
    # Run the ffmpeg command
    subprocess.run(command, check=True)

# Example usage
video_path = '/home/brownjordan317/fall_2024/CSCI443/Github/CSCI-443/object_detection/videoplayback.mp4'  # Path to your video file
output_folder = 'object_detection/video_frames_every_15'  # Folder to store extracted images
video_to_images(video_path, output_folder, image_format='png', fps=1/15)  # Extract 1 frame per second
