# -*- coding: utf-8 -*-


from imutils import face_utils
from imutils.video import VideoStream
import imutils
import time
import dlib
import cv2
import numpy as np
from scipy.spatial import distance 
import os

landmark_predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
landmark_detect = dlib.get_frontal_face_detector()

(smile_start,smile_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
def detect_lips(lip):
    corner_A = distance.euclidean(lip[3],lip[9])
    corner_B = distance.euclidean(lip[2], lip[10])
    corner_C = distance.euclidean(lip[4], lip[8])
    avg = (corner_A+corner_B+corner_C)/3
    corner_D = distance.euclidean(lip[0], lip[6])
    ratio=avg/corner_D
    return ratio

webcam = VideoStream(src=0).start()
try:
      
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
    
while True:
    window_frame = webcam.read()
    window_frame = imutils.resize(window_frame, width=450)
    gray = cv2.cvtColor(window_frame, cv2.COLOR_BGR2GRAY)
    anchor = landmark_detect(gray, 0)
    for box in anchor:
        smile_finder = landmark_predict(gray, box)
        smile_finder = face_utils.shape_to_np(smile_finder)
        smile= smile_finder[smile_start:smile_end]
        ratio= detect_lips(smile)
        smileHull = cv2.convexHull(smile)
        cv2.drawContours(window_frame, [smileHull], -1, (255, 0, 0), 1)
        count = 0
        tot = 0
        if ratio <= .2 or ratio > .25 :
            count = count+1
        else:
                tot= tot+1
                window_frame = webcam.read()
                time.sleep(.3)
                name = './data/frame' + str(count) + '.jpg'
                print ('Creating...' + name)
                cv2.imwrite(name, window_frame)
                count = count+1
    cv2.imshow("Frame", window_frame)
    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break
webcam.stop()   
cv2.destroyAllWindows()
