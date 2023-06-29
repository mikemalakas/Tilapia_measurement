#Install scipy di CMD
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

import datetime
import time
import os
import pandas as pd
from blynklib import Blynk

#Import Numpy
#Install Numpy di CMD
import numpy as np

#Install Imutills di CMD
import imutils

#Import OpenCV
import cv2

import pyrebase


config = { 
  "apiKey": "AIzaSyBAbFIdnN9K2FrMU9cbg6tuPuyJNCDu_go",
  "authDomain": "tilapiacam-3614d.firebaseapp.com",
  "projectId": "tilapiacam-3614d",
  "databaseURL": "https://tilapiacam-3614d-default-rtdb.asia-southeast1.firebasedatabase.app/",
  "storageBucket": "tilapiacam-3614d.appspot.com",
  "messagingSenderId": "752495443902",
  "appId": "1:752495443902:web:3a7b88c5b5466a708a03ef",
  "measurementId": "G-Z2STP7NT0F"}

firebase = pyrebase.initialize_app(config)
db = firebase.database()


#Initializing the variable "midpoint"
#Determining the midpoint of the object to be measured
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#Activate Cam
cap = cv2.VideoCapture(1)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

# Set the desired window size
window_width = 900
window_height = 700
cv2.resizeWindow('Camera', window_width, window_height)

# Initialize previous dimensions
previous_dimensions = None

# Generate a timestamp for the Excel file name
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
excel_filename = "dimensions_{}.xlsx".format(timestamp)


#Create an empty DataFrame 
dimensions_data = pd.DataFrame(columns=["Height (CM)", "Length (CM)",])

last_capture_time = 0  # Initialize last_capture_time outside the loop
image_counter = 1  # Initialize the image counter
captured_images = []  # List to store the captured image filenames

#calculate weight
def calculate_weight(length):
    weight = 0.0203 * length**3.0604
    return weight

weight = 0.0

#If the camera is active and the video has started, then run the program below
while (cap.read()):
        ref,frame = cap.read()
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        orig = frame[:1080,0:1920]
        
        #Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)

        result_img = closing.copy()
        contours,hierachy = cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        object_count = 0

        
        #Converting the pixel measurement values to centimeters
        pixelsPerMetric = None

        #Creating loop condition
        #Initializing the Variable "cnt" as "contours"
        for cnt in contours:

            #Reading the measured area of the object
            area = cv2.contourArea(cnt)

            #If the area is less than 1000 and greater than 12000 pixels,
            #then perform measurement
            if area < 1000 or area > 120000:
                continue


            #Calculating the bounding box of the object contours
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

            #Drawing the center point on the object
            cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)

            #Drawing a line on the center point
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

            #Calculating the Euclidean distance between the center points
            height_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            length_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            #If the variable "pixelsPerMetric" has not been initialized, then
            #Calculate it as the ratio of pixels to the provided metric
            #In this case, centimeters (CM)
            if pixelsPerMetric is None:
                pixelsPerMetric = height_pixel
                pixelsPerMetric = length_pixel
            height = height_pixel
            length = length_pixel

            #Depicting the size of an object in the image
            cv2.putText(orig, "H:{:.1f}CM".format(height_pixel/25.5),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            cv2.putText(orig, "L:{:.1f}CM".format(length_pixel/25.5),(int(tltrX - 55), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            #cv2.putText(orig,str(area),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2)
            weight = calculate_weight(length / 25.5)
            # Display the weight
            cv2.putText(orig, "Weight:{:.2f} g".format(weight), (int(tltrX + 65), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            object_count+=1
               
        #Display the number of object detected
        cv2.putText(orig, "object: {}".format(object_count),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)  
        cv2.imshow('Camera',orig)

        # Create a folder with a unique name based on the timestamp
        folder_name = "DATA_{}".format(timestamp)
        # Check if the folder already exists
        if not os.path.exists(folder_name):
            # Create the folder
            os.mkdir(folder_name)

        # Create the paths for the captured image and the Excel file inside the folder
        filename = "{}/captured_image{}.jpg".format(folder_name, image_counter)
        excel_filename = "{}/dimensions_{}.xlsx".format(folder_name, timestamp)
        
        #Automatic Capture when object detected
        if object_count > 0:
            current_time = time.time()

            #5sec delay to capture again
            if current_time - last_capture_time >= 10:

                #filename = "captured_image{}.jpg".format(image_counter)
                cv2.imwrite(filename, orig)
                print("Image captured: {}".format(filename))
                print("Dimensions of the captured object:")
                print("Height: {:.1f} CM".format(height / 25.5))
                print("Length: {:.1f} CM".format(length / 25.5))
                print("Weight: {:.2f} g".format(weight))

                # Calculate the dimensions in centimeters
                height_cm = height / 25.5
                length_cm = length / 25.5
                weight = calculate_weight(length_cm)

                # Create data dictionary for firebase
                data = {
                    #"height": height_cm,
                    "length": round(length_cm, 2),
                    "weight": round(weight, 2)
                        }
                # Send data to the database
                db.child("dimension").update(data)

                # Create a DataFrame for the new dimensions
                new_data = pd.DataFrame({"Height (CM)": [height_cm], "Length (CM)": [length_cm], "Weight (g)": [weight] })

                #Concatenate the new data with the existing dimensions_data
                dimensions_data = pd.concat([dimensions_data, new_data], ignore_index=True)

                #update last capture time and increment image counter
                last_capture_time = current_time
                image_counter += 1

                # Add the captured image filename to the list
                captured_images.append(filename)  

        #Press ESC to exit
        key = cv2.waitKey(1) 
        if key == 27:
            break

        #Press C to capture image    
        if key == ord('c'):
            cv2.imwrite("{}/captured_image{}.jpg".format(folder_name, image_counter), orig)
            print("Image captured!")

            # Print the dimensions of the captured object
            print("Dimensions of the captured object:")
            print("Height: {:.1f} CM".format(height / 25.5))
            print("Length: {:.1f} CM".format(length / 25.5))
            print("Weight: {:.2f} grams".format(weight))

            #Create a DataFrame for the new dimensions
            new_data = pd.DataFrame({"Height (CM)": [height_cm], "Length (CM)": [length_cm], "Weight (g)": [weight]})

            #Concatenate the new data with the existing dimensions_data
            dimensions_data = pd.concat([dimensions_data, new_data], ignore_index=True)


        #Delete previous images and its data when press D
        if key == ord('d'):
            if captured_images:
                # Remove the last captured image and its data
                previous_image_filename = captured_images.pop()
                os.remove(previous_image_filename)
                print("Previous image deleted: {}".format(previous_image_filename))
                
                # Remove the previous row of data from the DataFrame
                dimensions_data = dimensions_data[:-1]
            
                print("Previous row deleted.")
            else:
                print("No previous image to delete.")  
      
# Save the updated DataFrame to the Excel file
dimensions_data.to_excel(excel_filename, index=False)

cap.release()
cv2.destroyAllWindows()