import cv2
import numpy as np

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

img_no=0

train=[]

filename=input()
while 1:
    
    ret,frame=cap.read()
    if ret==False:
        continue
        
    faces=face_cascade.detectMultiScale(frame,1.3,4)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,150),3)
        
        offset=0
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        img_no+=1
        if img_no%10==0:
            train.append(face_section)
            print(len(train))

        
    cv2.imshow("Fewery",frame)
    
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
        
train=np.asarray(train)
train=train.reshape((train.shape[0],-1))

np.save("train_data/"+filename+".npy",train)

print(train.shape)
print("Saved images at:"+"train_data/"+filename+".npy")
        
cap.release()
cv2.destroyAllWindows()