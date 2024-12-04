# image_corrector/image_processing.py
import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # 左上角
    rect[2] = pts[np.argmax(s)]       # 右下角
    rect[1] = pts[np.argmin(diff)]    # 右上角
    rect[3] = pts[np.argmax(diff)]    # 左下角

    return rect

def four_point_transform(image, pts):
    if not isinstance(pts, np.ndarray):
        raise TypeError("pts必须是numpy数组")
    if pts.shape != (4, 2):
        raise ValueError("pts的形状必须是 (4, 2)")
    if pts.dtype != np.float32 and pts.dtype != np.float64:
        raise TypeError("pts的类型必须是 float32 或 float64")

    rect = order_points(pts).astype("float32")
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 调试信息
    print(f"rect (src points):\n{rect}")
    print(f"dst (destination points):\n{dst}")

    # 确保rect和dst都是float32
    rect = rect.astype(np.float32)
    dst = dst.astype(np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    print(f"透视变换矩阵 (M):\n{M}")

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def preprocess_image(image, user_points=None):
    if image is None:
        raise ValueError("preprocess_image函数接收到None作为输入")
    
    if user_points is not None:
        pts = np.array(user_points, dtype="float32")
    else:
        # 自动检测角点
        pts = detect_edges_and_find_contours(image)
    
    print(f"使用的角点:\n{pts}")

    warped = four_point_transform(image, pts)
    print("透视变换完成")

    # warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # warped_gray = cv2.medianBlur(warped_gray, 3)
    # _, warped_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("图像预处理完成")
    # return warped_thresh
    return warped

def detect_edges_and_find_contours(image):
    if image is None:
        raise ValueError("detect_edges_and_find_contours函数接收到None作为输入")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("未找到任何轮廓")

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            print("找到四边形轮廓")
            return approx.reshape(4, 2)

    h, w = image.shape[:2]
    print("未找到四边形轮廓，使用图像四角")
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")

def correct_image(image_path, output_path, user_points=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    print(f"读取图像成功: {image_path}, 图像尺寸: {image.shape}")

    preprocessed = preprocess_image(image, user_points)
    cv2.imwrite(output_path, preprocessed)
    print(f"保存处理后的图像: {output_path}")