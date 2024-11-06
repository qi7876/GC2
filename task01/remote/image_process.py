import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template_string
import os


def apply_sepia(image):
    """
    Add image fading effect
    """
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    
    sepia_image = cv2.transform(image, sepia_filter)
    
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image


def add_noise(image):
    """
    Add noise to picture
    """
    noise_intensity = 25  # Adjust the noise intensity.
    noise = np.random.normal(0, noise_intensity, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def apply_blur(image):
    """
    Apply blur effect to image
    """
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image


def adjust_brightness_contrast(image, brightness=-30, contrast=50):
    """
    Adjust the brightness and contrast of image
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)
    return adjusted_image


def process_image(image):
    """
    Process image, add the old movie effect.
    """
    sepia_image = apply_sepia(image)
    # noisy_image = add_noise(sepia_image)
    noisy_image = sepia_image
    blurred_image = apply_blur(noisy_image)
    final_image = adjust_brightness_contrast(blurred_image) 
    
    return final_image


def process_single_image(image_path, output_dir):
    """
    Process for a single image
    """
    image = cv2.imread(image_path)
    # Apply the old movie effect.
    processed_image = process_image(image)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, processed_image)


def process_video(video_path, output_dir):
    """
    Process every frame in a video, and save the result as a new video.
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_image(frame)
        out.write(processed_frame)

    # Release the memory.
    cap.release()
    out.release()


def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            print(file)
            file_path = os.path.join(root, file)
            
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                process_single_image(file_path, output_dir)
            elif file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                process_video(file_path, output_dir)

if __name__ == "__main__":
    os.chdir("/root/projects/GC2/task/task01")
    input_dir = "./resource/"
    output_dir = "./processed/"
    try:
        process_directory(input_dir, output_dir)
        print("Complete!")
    except Exception as e:
        print("Error:{1}".format(e))