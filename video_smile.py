import cv2
import csv
from os import listdir

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

#select number of cascades
TRIAL_NUM = 3

if TRIAL_NUM == 1:
  cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
  param_scale = 1.1
  param_neighbors = 3
elif TRIAL_NUM == 2:
  cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
  param_scale = 1.3
  param_neighbors = 10
elif TRIAL_NUM == 3:
  cascade_path = 'cascade_w20_h10_numStages20_numPos1000.xml'
  param_scale = 1.1
  param_neighbors = 3
elif TRIAL_NUM == 4:
  cascade_path = 'cascade_w20_h10_numStages20_numPos1000.xml'
  param_scale = 1.1
  param_neighbors = 5

#Predict negative samples

# First, a smile cascade is instantiated. 
# Then, the default/custom cascade is loaded into the CascadeClassifier object depending on the TRIAL_NUM specified.

smile_cascade = cv2.CascadeClassifier()
smile_cascade.load(cascade_path)

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smile = smile_cascade.detectMultiScale(gray, param_scale, param_neighbors)

    for (x,y,w,h) in smile:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        faceROI = frame[y:y+h,x:x+w]
         # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()