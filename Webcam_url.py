from flask import Flask, render_template, Response
import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

app = Flask('hello')
cap = cv2.VideoCapture(1)

#calculate weight
def calculate_weight(length):
    weight = 0.0203 * length**3.0604
    return weight

#initialize midpoint of object
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((3, 3), np.uint8)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

            result_img = closing.copy()
            contours, hierachy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            object_count = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 1000 or area > 120000:
                    continue

                orig = frame.copy()
                box = cv2.minAreaRect(cnt)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)

                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)

                cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
                cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

                height_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                length_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                # Convert length and height to your desired unit of measurement
                length_cm = length_pixel/25.5
                height_cm = height_pixel/25.5
                weight = calculate_weight(length_cm)
                object_count += 1

            cv2.putText(orig, "Object Count: {}".format(object_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(orig, "L: {:.2f} cm".format(length_cm), (int(tltrX) - 60, int(tltrY) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(orig, "H: {:.2f} cm".format(height_cm), (int(trbrX), int(trbrY) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(orig, "W:{:.2f} g".format(weight), (int(tltrX + 80), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frame = cv2.resize(orig, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
            

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
            <div class="col-lg-8 offset-lg-2">
                <h3 class="mt-5">Live Streaming</h3>
                <img src="/video_feed" width="100%">
            </div>
        </div>
    </div>
    </body>        
    """

if __name__ == '__main__':
    app.run(debug=True, host = "192.168.1.8")
