# Display an image using OpenCV and Flask.
import cv2
import os
import base64
from flask import Flask, render_template_string

os.chdir("/root/projects/GC2/task/task01/")
app = Flask(__name__)

def get_image_base64(image_path):
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return image_base64

@app.route('/')
def index():
    image_path = './resource/macintosh.jpg'  # Path to image.
    
    image_base64 = get_image_base64(image_path)
    
    html_content = f'''
    <html>
    <head><title>Image Display</title></head>
    <body>
        <h1>Hello, Flask!</h1>
        <img src="data:image/jpeg;base64,{image_base64}" alt="Image">
    </body>
    </html>
    '''
    
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)