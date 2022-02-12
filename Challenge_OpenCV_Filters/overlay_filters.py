import cv2
import numpy as np
import os

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eyes_cascade=cv2.CascadeClassifier("frontalEyes35x16.xml")  
nose_cascade=cv2.CascadeClassifier("Nose18x15.xml")

specs=cv2.imread("glasses.png",-1)                               #-1 is to read image in BGRA channel (-1 gives A (transparency channel)
stash=cv2.imread("mustache.png",-1)

while 1:
    ret,frame=cap.read()           
    if ret==False:
        continue
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
    
    faces=face_cascade.detectMultiScale(frame,1.5,5)
    
    for fx,fy,fw,fh in faces:
        roi_face=frame[fy:fy+fh,fx:fx+fw]
        
        eyes=eyes_cascade.detectMultiScale(roi_face,1.5,5)
        noses=nose_cascade.detectMultiScale(roi_face,1.5,5)
    
        eyes=sorted(eyes,key=lambda e:e[2]*e[3])                        #e is a tuple, e[2]=w, e[3]=h
        noses=sorted(noses,key=lambda n:n[2]*n[3])                      #similarily
        
        for x,y,w,h in eyes[-1:]:
            specs2=cv2.resize(specs.copy(),(w,h))
            
            #cv2.rectangle(frame,(fx+x,fy+y),(fx+x+w,fy+y+h),(255,100,150),5)
            
            for i in range(0,h):
                for j in range(0,w):
                    if specs2[i,j][3]!=0:
                        frame[fy+y+i,fx+x+j]=specs2[i,j]

        for x,y,w,h in noses[-1:]:
            stash2=cv2.resize(stash.copy(),(w,h))

            for i in range(0,h):
                for j in range(0,w):
                    if stash2[i,j][3]!=0:
                        frame[fy+y+int(h/2)+i,fx+x+j]=stash2[i,j]
                    
    cv2.imshow("Fewery",frame)
    
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
