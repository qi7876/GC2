# utils.py
import cv2
from PIL import Image, ImageTk


def cv2_to_tk(image):
    # 将OpenCV图像转换为Tkinter兼容的图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    return ImageTk.PhotoImage(pil_image)


def resize_image(image, max_width, max_height):
    # 调整图像大小以适应指定的宽度和高度，同时保持纵横比
    height, width = image.shape[:2]
    ratio = min(max_width / width, max_height / height, 1)
    new_size = (int(width * ratio), int(height * ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image, ratio
