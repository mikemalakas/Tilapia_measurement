import cv2
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response
import pyrebase
import time
from ultralytics import YOLO

app = Flask('hello')

config = {
"apiKey": "AIzaSyBAbFIdnN9K2FrMU9cbg6tuPuyJNCDu_go",
  "authDomain": "tilapiacam-3614d.firebaseapp.com",
  "projectId": "tilapiacam-3614d",
  "databaseURL": "https://tilapiacam-3614d-default-rtdb.asia-southeast1.firebasedatabase.app/",
  "storageBucket": "tilapiacam-3614d.appspot.com",
  "messagingSenderId": "752495443902",
  "appId": "1:752495443902:web:3a7b88c5b5466a708a03ef",
  "measurementId": "G-Z2STP7NT0F"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

model = YOLO('besty.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

def calculate_weight(length):
    weight = 0.0203 * length ** 3.0604
    return weight


cap = cv2.VideoCapture(1)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


def gen_frames():
    count = 0
    last_capture_time = 0
    object_length = 0
    object_count = 0    
    while True:
        ret, frame = cap.read()
        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))

        results = model.predict(frame)
        a = results[0].boxes.boxes
        px = pd.DataFrame(a).astype("float")

        object_count = len(px)
        count_text = "Object Count: {}".format(object_count)
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

            object_length = np.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2))
            object_cm = object_length / 25.5
            weight = 0.0203 * object_cm ** 3.0604
            length_text = "L: {:.2f}cm".format(object_cm)
            weight_text = "W: {:.2f}g".format(weight)
            cv2.putText(frame, length_text, (x1 + 90, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, weight_text, (x1 + 250, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

        if object_count > 0:
            current_time = time.time()

            if current_time - last_capture_time >= 10:
                data = {
                    "length": round(object_cm, 2),
                    "weight": round(weight, 2)
                }
                db.child("dimension").update(data)
                last_capture_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
        <body>
        <div class="container">
            <div class="row">
                <div class="col-lg-8  offset-lg-2">
                    <h3 class="mt-5">Live Streaming</h3>
                    <img src="/video_feed" width="100%">
                </div>
            </div>
        </div>
        </body>        
    """

if __name__ == '__main__':
    app.run()
