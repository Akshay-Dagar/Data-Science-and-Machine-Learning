import cv2
import numpy as np
import os
import pandas as pd

cap=cv2.VideoCapture(0)

eyes_cascade=cv2.CascadeClassifier("frontalEyes35x16.xml")  
nose_cascade=cv2.CascadeClassifier("Nose18x15.xml")

specs=cv2.imread("glasses.png",-1)                               #-1 is to read image in BGRA channel (-1 gives A (transparency channel)
stash=cv2.imread("mustache.png",-1)

imp=cv2.imread("Before.png")
imp=cv2.cvtColor(imp,cv2.COLOR_BGR2BGRA)

eyes=eyes_cascade.detectMultiScale(imp,1.5,5)
nose=nose_cascade.detectMultiScale(imp,1.5,5)

for x,y,w,h in eyes:
    specs2=cv2.resize(specs.copy(),(w,h))
    
    for i in range(0,h):
        for j in range(0,w):
            if specs2[i,j][3]!=0:
                imp[y+i,x+j]=specs2[i,j]
                    
for x,y,w,h in nose:
    stash2=cv2.resize(stash.copy(),(w,h))
    
    for i in range(0,h):
        for j in range(0,w):
            if stash2[i,j][3]!=0:
                imp[y+int(h/2)+i,x+j]=stash2[i,j]
                    
cv2.imshow("Fewery",imp)

imp=np.reshape(imp,(imp.shape[0]*imp.shape[1],imp.shape[2]))

d={"Channel 1":imp[:,0],
   "Channel 2":imp[:,1],
   "Channel 3":imp[:,2]}
df=pd.DataFrame(d)
df.to_csv("submission.csv",index=False)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
