import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template_string

app = Flask(__name__)

video_feed_url = 'http://localhost:4500/video_feed'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_frame_from_stream():
    stream = requests.get(video_feed_url, stream=True)
    bytes_data = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield img

def generate_original():
    for frame in get_frame_from_stream():
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_with_faces():
    for frame in get_frame_from_stream():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/original_feed')
def original_feed():
    return Response(generate_original(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_feed')
def face_feed():
    return Response(generate_with_faces(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>Face Detection</title>
    </head>
    <body>
        <h1>Live Video Stream</h1>
        <div style="display: flex;">
            <div style="flex: 1; margin-right: 10px;">
                <h2>Original Stream</h2>
                <img src="{{ url_for('original_feed') }}" width="100%">
            </div>
            <div style="flex: 1;">
                <h2>Face Detection Stream</h2>
                <img src="{{ url_for('face_feed') }}" width="100%">
            </div>
        </div>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
