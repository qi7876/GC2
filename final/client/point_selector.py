# point_selector.py
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

class DraggablePoint:
    def __init__(self, canvas, x, y, index, callback):
        self.canvas = canvas
        self.index = index
        self.callback = callback
        self.radius = 5
        self.tag = f"point_{index}"
        self.id = canvas.create_oval(x - self.radius, y - self.radius,
                                     x + self.radius, y + self.radius,
                                     fill='red', outline='black', tags=self.tag)
        self.text_id = canvas.create_text(x, y - 10, text=str(index + 1), fill='yellow', tags=self.tag)
        self.drag_data = {"x": 0, "y": 0}

        # Bind events for dragging
        canvas.tag_bind(self.tag, "<ButtonPress-1>", self.on_press)
        canvas.tag_bind(self.tag, "<ButtonRelease-1>", self.on_release)
        canvas.tag_bind(self.tag, "<B1-Motion>", self.on_motion)

    def on_press(self, event):
        # Record the initial position
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_release(self, event):
        # Reset drag data
        self.drag_data["x"] = 0
        self.drag_data["y"] = 0
        self.callback()

    def on_motion(self, event):
        # Calculate the distance moved
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        # Move the object the appropriate amount
        self.canvas.move(self.id, dx, dy)
        self.canvas.move(self.text_id, dx, dy)
        # Record the new position
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        # Update the point position
        self.callback()

    def get_position(self):
        coords = self.canvas.coords(self.id)
        x = (coords[0] + coords[2]) / 2
        y = (coords[1] + coords[3]) / 2
        return [x, y]

class PointSelector:
    def __init__(self, master, image_path, max_width=800, max_height=600):
        self.master = master
        self.master.title("选择四个角点并调整位置")
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        self.original_height, self.original_width = self.original_image.shape[:2]
        self.scaled_image, self.scale = self.scale_image(self.original_image, max_width, max_height)
        self.points = []
        self.draggable_points = []
        self.lines = []
        self.max_points = 4
        self.current_index = 0

        # Convert OpenCV image (BGR) to PIL image (RGB)
        image_rgb = cv2.cvtColor(self.scaled_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Create Canvas
        self.canvas = tk.Canvas(master, width=self.scaled_image.shape[1], height=self.scaled_image.shape[0])
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Prompt Label
        self.label = tk.Label(master, text="正在自动检测角点...")
        self.label.pack(pady=10)

        # Done Button
        self.done_button = tk.Button(master, text="完成", command=self.on_done, state=tk.DISABLED)
        self.done_button.pack(pady=10)

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.on_click)

        # Automatically detect corner points
        self.auto_detect_corners()

    def scale_image(self, image, max_width, max_height):
        height, width = image.shape[:2]
        ratio = min(max_width / width, max_height / height, 1)
        new_size = (int(width * ratio), int(height * ratio))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image, ratio

    def auto_detect_corners(self):
        try:
            pts = self.detect_edges_and_find_contours(self.original_image)
            # Scale points
            scaled_pts = pts * self.scale
            for i, (x, y) in enumerate(scaled_pts):
                point = DraggablePoint(self.canvas, x, y, i, self.update_points)
                self.draggable_points.append(point)
                self.points.append(point.get_position())
                self.current_index += 1
            self.done_button.config(state=tk.NORMAL)
            self.label.config(text="自动检测到四个角点。您可以拖动它们以调整位置。")
            self.draw_lines()
        except Exception as e:
            self.label.config(text="自动检测角点失败，请手动选择角点。")
            print(f"自动检测角点失败: {e}")

    def on_click(self, event):
        if self.current_index < self.max_points:
            x, y = event.x, event.y
            point = DraggablePoint(self.canvas, x, y, self.current_index, self.update_points)
            self.draggable_points.append(point)
            self.points.append(point.get_position())
            self.current_index += 1
            if self.current_index == self.max_points:
                self.done_button.config(state=tk.NORMAL)
                self.label.config(text="已选择四个角点。您可以拖动它们以调整位置。")
                self.draw_lines()
        else:
            self.label.config(text="已选择四个角点。您可以拖动它们以调整位置。")

    def update_points(self):
        # Update the points list based on draggable points' current positions
        self.points = [p.get_position() for p in self.draggable_points]
        self.update_lines()

    def on_done(self):
        if len(self.points) == self.max_points:
            self.master.destroy()

    def get_points(self):
        # Map scaled points back to original image coordinates
        return [ [x / self.scale, y / self.scale] for x, y in self.points ]

    def detect_edges_and_find_contours(self, image):
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

    def draw_lines(self):
        # Draw lines between points in order: 0-1-2-3-0
        if len(self.points) < 4:
            return
        self.lines = []
        for i in range(self.max_points):
            start = self.points[i]
            end = self.points[(i + 1) % self.max_points]
            line = self.canvas.create_line(start[0], start[1], end[0], end[1], fill='blue', width=2)
            self.lines.append(line)

    def update_lines(self):
        # Update the position of the lines based on current points
        for i in range(len(self.lines)):
            start = self.points[i]
            end = self.points[(i + 1) % len(self.points)]
            self.canvas.coords(self.lines[i], start[0], start[1], end[0], end[1])