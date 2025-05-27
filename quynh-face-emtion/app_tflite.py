
from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from tensorflow.keras.utils import img_to_array

app = Flask(__name__)

# Load mô hình TFLite
interpreter = tflite.Interpreter(model_path="model.h5")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load cascade khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Nhãn cảm xúc cần nhận diện
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
selected_indices = [0, 3, 4, 6]  # mapping theo model 7 lớp gốc

HTML = """
<html>
<head><title>Emotion Detection</title></head>
<body style="text-align: center;">
    <h2>Emotion Detection (TFLite + Flask)</h2>
    <img src="{{ url_for('video_feed') }}" width="640" height="480">
</body>
</html>
"""

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                interpreter.set_tensor(input_details[0]['index'], roi)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                # Lấy cảm xúc từ 4 nhãn chọn lọc
                filtered = [output_data[i] for i in selected_indices]
                label = emotion_labels[np.argmax(filtered)]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
