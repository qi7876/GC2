import cv2
import threading
from flask import Flask, Response

app = Flask(__name__)

video_capture = None

def capture_frames():
    global video_capture
    video_capture = cv2.VideoCapture(0)

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate():
    global video_capture
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=capture_frames).start()
    app.run(host='0.0.0.0', port=4500)