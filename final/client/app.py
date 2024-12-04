import os
import shutil
import threading
from tkinter import StringVar, Frame, Label, Entry, Button, LEFT, Checkbutton, W, E, Toplevel, messagebox, DISABLED, NORMAL, filedialog, scrolledtext, BOTH, END, Tk, WORD, BooleanVar
from PIL import Image as PILImage, ImageTk
import requests
from datetime import datetime

from openai import OpenAI
import google.generativeai as genai

from image_corrector import correct_image
from remote import upload_images, get_processed_text
from point_selector import PointSelector


class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("💯大学生考试周小助手💯")
        self.master.geometry("800x600")

        # 初始化变量
        self.server_ip = StringVar(value="localhost")
        self.server_port = StringVar(value="8000")
        self.selected_images = []
        self.result_dir = "result"
        self.output_text_path = os.path.join(self.result_dir, "output_text.txt")
        self.output_mermaid_text_path = os.path.join(self.result_dir, "mermaid.txt")
        self.output_mermaid_image_path = os.path.join(self.result_dir, "diagram.png")
        self.mermaid_url = "https://kroki.io/mermaid/png"
        self.enable_correction = BooleanVar(value=True)
        self.archive_dir = "archive"

        # 创建缓存目录
        os.makedirs(self.result_dir, exist_ok=True)

        self.create_widgets()

    def create_widgets(self):
        server_frame = Frame(self.master)
        server_frame.pack(pady=10)

        Label(server_frame, text="💻服务器IP:").grid(
            row=0, column=0, padx=5, pady=5, sticky=E
        )
        self.ip_entry = Entry(server_frame, textvariable=self.server_ip)
        self.ip_entry.grid(row=0, column=1, padx=5, pady=5)

        Label(server_frame, text="💻服务器端口:").grid(
            row=0, column=2, padx=5, pady=5, sticky=E
        )
        self.port_entry = Entry(server_frame, textvariable=self.server_port)
        self.port_entry.grid(row=0, column=3, padx=5, pady=5)

        correction_frame = Frame(self.master)
        correction_frame.pack(pady=5)

        self.correction_checkbox = Checkbutton(
            correction_frame, text="🖼️开启图像矫正功能", variable=self.enable_correction
        )
        self.correction_checkbox.pack(anchor=W, padx=10)

        select_frame = Frame(self.master)
        select_frame.pack(pady=10)

        self.select_button = Button(
            select_frame, text="☑️选择图片", command=self.select_images
        )
        self.select_button.pack(side=LEFT, padx=5)

        self.image_count_label = Label(select_frame, text="️已选择 0 张图片")
        self.image_count_label.pack(side=LEFT, padx=5)

        # 开始处理按钮
        process_frame = Frame(self.master)
        process_frame.pack(pady=10)

        self.process_button = Button(
            process_frame, text="⚙️开始处理", command=self.start_processing
        )
        self.process_button.pack()

        # 历史记录显示
        history_frame = Frame(self.master)
        history_frame.pack(pady=10, fill=BOTH, expand=True)

        Label(history_frame, text="📹运行记录:").pack(anchor=W)

        self.history_text = scrolledtext.ScrolledText(
            history_frame, height=15, state="disabled"
        )
        self.history_text.pack(fill=BOTH, expand=True)

    def select_images(self):
        try:
            files = filedialog.askopenfilenames(
                title="☑️选择图片",
                filetypes=[
                    ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")
                ],  # NOTE: macOS上必须使用空格分割
            )
            if files:
                # 打印选中的文件路径以进行调试
                print("📁选中的文件:", files)
                self.selected_images = list(files)
                self.image_count_label.config(
                    text=f"️已选择 {len(self.selected_images)} 张图片"
                )
                self.update_history(f"☑️选择了 {len(self.selected_images)} 张图片")
            else:
                self.update_history("☑️未选择任何图片")
        except Exception as e:
            messagebox.showerror("⚠️错误", f"选择图片时出错: {str(e)}")
            self.update_history(f"⚠️错误: {str(e)}")

    def start_processing(self):
        if not self.selected_images:
            messagebox.showwarning("⚠️警告", "请至少选择一张图片。")
            return

        # 禁用按钮以防重复点击
        self.process_button.config(state=DISABLED)
        self.select_button.config(state=DISABLED)

        processing_thread = threading.Thread(target=self.process_images)
        processing_thread.start()

    def process_images(self):
        try:
            server_url = (
                f"http://{self.server_ip.get()}:{self.server_port.get()}/upload-images/"
            )
            self.update_history(f"💻服务器URL: {server_url}")

            processed_image_paths = []
            processed_text = []

            for idx, image_path in enumerate(self.selected_images, start=1):
                self.update_history(
                    f"⚙️正在处理图片 {idx}/{len(self.selected_images)}: {os.path.basename(image_path)}"
                )

                # 定义输出图片路径
                output_image_path = os.path.join(self.result_dir, f"{idx}.jpg")
                self.update_history(f"📁输出图片路径: {output_image_path}")

                try:
                    if self.enable_correction.get():
                        # 弹出角点选择窗口
                        user_points = self.select_points(image_path)
                        if user_points is None or len(user_points) != 4:
                            self.update_history(
                                f"🏃未选择足够的角点，跳过图片: {image_path}"
                            )
                            continue

                        # 图像矫正与预处理
                        correct_image(image_path, output_image_path, user_points)
                        processed_image_paths.append(output_image_path)
                        self.update_history(f"✅图像矫正完成: {output_image_path}")
                    else:
                        # 如果不启用矫正，直接复制原始图片到缓存目录
                        shutil.copy2(image_path, output_image_path)
                        processed_image_paths.append(output_image_path)
                        self.update_history(f"✅已复制原始图片: {output_image_path}")
                except Exception as e:
                    self.update_history(f"⚠️处理失败: {image_path}, 错误: {str(e)}")
                    continue

            if not processed_image_paths:
                self.update_history("🏃未成功处理任何图片，跳过后续处理。")
                return

            # 批量上传所有处理后的图片
            try:
                self.update_history(
                    f"⬆️正在上传 {len(processed_image_paths)} 张图片到服务器..."
                )
                response = upload_images(server_url, processed_image_paths)
                self.update_history(
                    "📷正在进行OCR识别，获取文本中..."
                )
                processed_text = get_processed_text(response)
                self.update_history("✅上传成功，获取文本完成。")
            except Exception as e:
                self.update_history(f"⚠️批量上传或获取文本失败: 错误: {str(e)}")
                return

            if not processed_text:
                self.update_history("⚠️服务器未返回任何文本，请检查图片内容。如果您开启了图片矫正，请确保框选到了正确的区域。")
                return

            with open(self.output_text_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            self.update_history(f"✅文本已保存: {self.output_text_path}")

            try:
                self.update_history("🐶智能助手正在为您生成思维导图...")
                model_response_mermaid = self.call_large_model_api_mermaid(
                    processed_text
                )
                with open(self.output_mermaid_text_path, "a", encoding="utf-8") as f:
                    f.write(model_response_mermaid)

                params = {"scale": 10}
                response = requests.post(
                    self.mermaid_url,
                    data=model_response_mermaid.encode("utf-8"),
                    headers={"Content-Type": "text/plain"},
                    params=params,
                )

                if response.status_code == 200:
                    with open(self.output_mermaid_image_path, "wb") as img_file:
                        img_file.write(response.content)
                    self.update_history("🐶思维导图绘制完成")
                else:
                    self.update_history(f"⚠️无法绘制思维导图: {response.status_code}")

                self.display_image(self.output_mermaid_image_path)

                self.update_history("🐶智能助手正在为您起草学习指导...")
                model_response_learnLM = self.call_large_model_api_LearnLM(
                    processed_text
                )
                with open(self.output_text_path, "a", encoding="utf-8") as f:
                    f.write("\n\n=== 智能助手响应 ===\n")
                    f.write(model_response_learnLM)
                self.update_history("🐶智能助手工作完成，结果已为您保存到本地。")
            except Exception as e:
                self.update_history(f"⚠️智能助手运行错误: {str(e)}")
                return

            self.show_model_response(model_response_learnLM)

            self.update_history("✅运行完成")

        except Exception as e:
            messagebox.showerror("⚠️错误", str(e))
            self.update_history(f"⚠️错误: {str(e)}")
        finally:
            self.clean_cache()
            self.process_button.config(state=NORMAL)
            self.select_button.config(state=NORMAL)

            self.selected_images = []
            self.image_count_label.config(text="️已选择 0 张图片")

    def select_points(self, image_path):
        selector_window = Toplevel(self.master)
        selector_window.grab_set()

        point_selector = PointSelector(selector_window, image_path)
        self.master.wait_window(selector_window)

        return point_selector.get_points()

    def clean_cache(self):
        try:
            if os.path.exists(self.result_dir) and os.listdir(self.result_dir):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_subdir = os.path.join(self.archive_dir, timestamp)
                os.makedirs(archive_subdir, exist_ok=True)
                self.update_history(f"📃正在将缓存内容备份到归档目录: {archive_subdir}")

                for item in os.listdir(self.result_dir):
                    s = os.path.join(self.result_dir, item)
                    d = os.path.join(archive_subdir, item)
                    shutil.copy2(s, d)
                self.update_history("✅缓存内容备份完成。")
            else:
                self.update_history("🈳缓存目录为空，无需备份。")

            if os.path.exists(self.result_dir):
                shutil.rmtree(self.result_dir)
            os.makedirs(self.result_dir, exist_ok=True)
            self.update_history("🧹缓存目录已清理。")
        except Exception as e:
            messagebox.showerror("⚠️错误", f"清理缓存时出错: {str(e)}")
            self.update_history(f"⚠️清理缓存时出错: {str(e)}")

    def call_large_model_api_mermaid(self, text):
        client = OpenAI(
            api_key="sk-87de969e77fb4967b5e80a9c0856dd1e",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {
                    "role": "system",
                    "content": "You are an educational expert LLM. Please organize and analyze the given text, identifying all the key knowledge points it contains. Using these knowledge points, generate a mind map in Chinese. The mind map should be in the Mermaid format to facilitate subsequent visualization. Only return the Mermaid code for the mind map, without any extra content, to avoid errors in the visualization step. Do not add ```mermaid ``` in your response.",
                },
                {"role": "user", "content": text},
            ],
        )

        return completion.choices[0].message.content

    def call_large_model_api_LearnLM(self, text):
        genai.configure(api_key="AIzaSyCJsrW4DEDJASQK7__cI3DRSYZUSVZLCbU")

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="learnlm-1.5-pro-experimental",
            generation_config=generation_config,
            system_instruction="You are an educational expert LLM. Please organize and analyze the given text, extracting the key knowledge points it contains. Based on these knowledge points, generate a study guide in Chinese. Do not include any prompts for further questions, as this conversation will only occur once. All content in your response should be in Chinese.",
        )

        chat_session = model.start_chat(history=[])

        response = chat_session.send_message(text)

        return response.text


    def display_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"⚠️文件未找到: {image_path}")

            pil_image = PILImage.open(image_path)
            original_width, original_height = pil_image.size

            window = Toplevel()
            window.title("🐶智能导图")

            window.geometry(
                f"{original_width + 20}x{original_height + 60}"
            )

            tk_image = ImageTk.PhotoImage(pil_image)

            label = Label(window, image=tk_image)
            label.image = tk_image  # 保持引用
            label.pack(padx=10, pady=10, expand=True)

            close_button = Button(window, text="❌关闭", command=window.destroy)
            close_button.pack(pady=(0, 10))

        except FileNotFoundError as fnf_error:
            messagebox.showerror("⚠️文件未找到", str(fnf_error))
        except IOError as io_error:
            messagebox.showerror(
                "⚠️图片打开失败", f"无法打开图片文件: {image_path}\n错误: {str(io_error)}"
            )
        except Exception as e:
            messagebox.showerror("⚠️错误", f"发生了一个未知错误:\n{str(e)}")

    def show_model_response(self, text):
        response_window = Toplevel(self.master)
        response_window.title("🐶智能助手")

        response_text = scrolledtext.ScrolledText(
            response_window, wrap=WORD, width=80, height=20
        )
        response_text.pack(padx=10, pady=10)
        response_text.insert(END, text)
        response_text.config(state="disabled")

        button_frame = Frame(response_window)
        button_frame.pack(pady=10)

        copy_button = Button(
            button_frame, text="⌨️复制内容", command=lambda: self.copy_to_clipboard(text)
        )
        copy_button.pack(side=LEFT, padx=5)

        close_button = Button(
            button_frame,
            text="❌关闭",
            command=lambda: response_window.destroy(),
        )
        close_button.pack(side=LEFT, padx=5)

    def copy_to_clipboard(self, text):
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        messagebox.showinfo("⌨️复制成功", "内容已复制到剪贴板。")

    def update_history(self, message):
        self.history_text.config(state="normal")
        self.history_text.insert(END, f"{message}\n")
        self.history_text.see(END)
        self.history_text.config(state="disabled")
        print(message)


def main():
    root = Tk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
