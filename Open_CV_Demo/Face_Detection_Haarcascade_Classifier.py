import cv2

cap=cv2.VideoCapture(0)

face_detector=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


while 1:
    
    ret,frame=cap.read()
    
    if ret==False:
        continue
        
    faces=face_detector.detectMultiScale(frame,1.3,3)
    
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,112),5)
        
    cv2.imshow("Fewery",frame)
    
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
cap.release()