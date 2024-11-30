# image_processing.py
import cv2
import numpy as np


def order_points(pts):
    # 初始化矩形（四个顶点）
    rect = np.zeros((4, 2), dtype="float32")

    # 计算点的和与差
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # 获取对应的顶点
    rect[0] = pts[np.argmin(s)]  # 左上角
    rect[2] = pts[np.argmax(s)]  # 右下角
    rect[1] = pts[np.argmin(diff)]  # 右上角
    rect[3] = pts[np.argmax(diff)]  # 左下角

    return rect


def four_point_transform(image, pts):
    # 获取排列后的顶点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新的宽度和高度
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # 构建变换后的目标坐标
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def preprocess_image(image):
    # 图像预处理
    warped_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.medianBlur(warped_gray, 3)
    _, warped_thresh = cv2.threshold(
        warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return warped_thresh
