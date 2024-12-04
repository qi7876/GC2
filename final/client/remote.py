# remote.py
import requests
import os

def upload_images(url, image_paths):
    files = []
    file_handlers = []
    try:
        for path in image_paths:
            if os.path.exists(path):
                file_handle = open(path, 'rb')
                file_handlers.append(file_handle)
                files.append(('files', (os.path.basename(path), file_handle, 'image/jpeg')))
            else:
                raise FileNotFoundError(f"文件未找到: {path}")

        response = requests.post(url, files=files)
        return response
    finally:
        for fh in file_handlers:
            fh.close()

def get_processed_text(response):
    if response.status_code == 200:
        data = response.json()
        return data.get("processed_text", [])
    else:
        raise ValueError(f"服务器返回错误: {response.text}")