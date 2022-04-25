import numpy as np
import cv2
import matplotlib.pyplot as plt
#import mediapipe as mp
cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default")
#lets do the cascade face detection then lets go for the media pipe human pose estimator
video = cv2.VideoCapture(0)
cv2.namedWindow('surya')
count = 0

while True:
    has_frame,frame = video.read()
    if not has_frame:
        print('not able to capture a photo')
    try:
        rgb =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #resize = cv2.resize(rgb,(400,400))
        detect = cascade.detectMultiScale(rgb,scaleFactor = 1.1,minNeighbors = 1)
        for (x,y,w,h) in detect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow(frame)
    except:
        break
    key = cv2.waitKey(1)
    if key%256 == 27:
        print('escape key pressed')
        break
    elif key%256 == 32:
        print('screen shot taken')
        name = 'surya_{}.png'.format(count)
        cv2.imwrite(name,frame)
    count+=1
video.release()
cv2.destroyWindow('surya')

#image = cv2.imread('./surya_152.png')
#manupulation(image)
#cv2.imshow(vankai)


