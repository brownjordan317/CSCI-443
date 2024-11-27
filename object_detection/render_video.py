import os
import subprocess
from natsort import natsorted

def images_to_video_ffmpeg(image_dir, output_video, fps=30):
    """
    Uses FFmpeg to convert .png images in a directory into a video.
    
    Args:
        image_dir (str): Path to the directory containing images.
        output_video (str): Path to save the output video file.
        fps (int): Frames per second for the video.
    """
    # Get all .png files in the directory, sorted numerically
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    images = natsorted(images)

    if not images:
        print("No .png images found in the directory.")
        return

    # Ensure a temporary directory exists for renaming
    temp_dir = os.path.join(image_dir, "temp_ffmpeg")
    os.makedirs(temp_dir, exist_ok=True)

    # Rename files to a sequence format: 0001.png, 0002.png, ...
    for idx, img in enumerate(images):
        old_path = os.path.join(image_dir, img)
        new_path = os.path.join(temp_dir, f"{idx:04d}.jpg")
        os.rename(old_path, new_path)

    # Generate the video using FFmpeg
    command = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "%04d.jpg"),  # Input file pattern
        "-c:v", "libx264",  # Codec
        "-pix_fmt", "yuv420p",  # Pixel format for compatibility
        output_video  # Output file
    ]

    # Run FFmpeg command
    try:
        subprocess.run(command, check=True)
        print(f"Video saved as {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running FFmpeg: {e}")
    finally:
        # Clean up the temporary directory
        for file in os.listdir(temp_dir):
            os.rename(os.path.join(temp_dir, file), os.path.join(image_dir, file))
        os.rmdir(temp_dir)

# Example usage
images_to_video_ffmpeg("object_detection/preds_30fps", "object_detection/preds_30fps.mp4")
