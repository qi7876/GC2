# gui.py
import cv2
import numpy as np
import tkinter
from PIL import Image, ImageTk
from image_processing import four_point_transform
from utils import cv2_to_tk, resize_image


class ImageCorrectorGUI:
    def __init__(self, master, image_path, on_confirm_callback):
        self.master = master
        self.master.title("图像矫正")
        self.on_confirm_callback = on_confirm_callback

        # 读取图像
        self.orig = cv2.imread(image_path)
        if self.orig is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 预处理图像以检测边框
        self.detected_pts = self.detect_edges_and_find_contours(self.orig.copy())

        # 调整图像以适应屏幕
        screen_width = self.master.winfo_screenwidth() - 100
        screen_height = self.master.winfo_screenheight() - 150
        self.display_image, self.ratio = resize_image(
            self.orig, screen_width, screen_height
        )
        self.display_pts = (self.detected_pts * self.ratio).tolist()

        # 创建Canvas
        self.canvas = tkinter.Canvas(
            master,
            width=self.display_image.shape[1],
            height=self.display_image.shape[0],
        )
        self.canvas.pack()

        # 将图像显示在Canvas上
        self.tk_image = cv2_to_tk(self.display_image)
        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor=tkinter.NW, image=self.tk_image
        )

        # 绘制四边形和角点
        self.point_radius = 8
        self.draw_polygon()

        # 绑定鼠标事件
        self.selected_point = None
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)

        # 添加确认按钮
        self.confirm_button = tkinter.Button(
            master, text="确认", command=self.on_confirm
        )
        self.confirm_button.pack(pady=10)

    def detect_edges_and_find_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # 查找轮廓
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # 遍历轮廓，寻找四边形
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                return approx.reshape(4, 2)

        # 如果没有找到四边形，使用图像的四个角
        h, w = image.shape[:2]
        return np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
        )

    def draw_polygon(self):
        # 删除之前的图形
        self.canvas.delete("polygon")
        self.canvas.delete("point")

        # 绘制四边形
        self.canvas.create_polygon(
            self.display_pts, outline="green", fill="", width=2, tags="polygon"
        )

        # 绘制角点
        for point in self.display_pts:
            self.canvas.create_oval(
                point[0] - self.point_radius,
                point[1] - self.point_radius,
                point[0] + self.point_radius,
                point[1] + self.point_radius,
                fill="red",
                outline="yellow",
                tags="point",
            )

    def on_button_press(self, event):
        x, y = event.x, event.y
        for i, point in enumerate(self.display_pts):
            px, py = point
            dist = np.hypot(px - x, py - y)
            if dist <= self.point_radius + 2:
                self.selected_point = i
                break

    def on_button_release(self, event):
        self.selected_point = None

    def on_mouse_move(self, event):
        if self.selected_point is not None:
            x = max(0, min(event.x, self.display_image.shape[1] - 1))
            y = max(0, min(event.y, self.display_image.shape[0] - 1))
            self.display_pts[self.selected_point] = [x, y]
            self.draw_polygon()

    def on_confirm(self):
        # 获取用户调整后的四点，并恢复到原始图像比例
        pts = np.array(self.display_pts) / self.ratio
        pts = pts.astype("float32")

        # 进行透视变换和预处理
        corrected_image = four_point_transform(self.orig, pts)
        preprocessed_image = self.on_confirm_callback(corrected_image)

        # 显示处理后的图像
        cv2.imshow("矫正后的图像", preprocessed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 关闭GUI
        self.master.destroy()
