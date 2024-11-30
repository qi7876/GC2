# main.py
import cv2
from tkinter import Tk
from image_processing import preprocess_image
from gui import ImageCorrectorGUI


def process_image(image):
    # 对矫正后的图像进行预处理
    preprocessed = preprocess_image(image)
    # 您可以在这里添加更多的处理步骤，例如保存图像或进行OCR
    cv2.imwrite("output.jpg", preprocessed)
    return preprocessed


def correct_image(image_path, image_num):
    # 创建Tkinter主窗口
    root = Tk()
    app = ImageCorrectorGUI(root, image_path, process_image)
    root.mainloop()

    # 加载并返回处理后的图像
    processed_image = cv2.imread("output.jpg")
    return processed_image


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    corrected_image = correct_image(image_path)
    print("图像矫正和预处理完成，保存为 'output.jpg'")
