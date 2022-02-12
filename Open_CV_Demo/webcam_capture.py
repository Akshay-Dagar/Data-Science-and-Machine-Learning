import cv2

cap=cv2.VideoCapture(0)       #specify device to be captured, we are capturing video through webcam with id 0

while 1:                      #we want this to capture infinitely until key 'q' is pressed from the keyboard
    
    ret,frame=cap.read()                
    #read the video from the webcam, frame is the image recorded, ret specfies whether or not the capture was succesful
    
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)        
    #getting grayscale image corresonding to the captured frame
    
    if ret==False:            #the capture may be unsuccesful due to reasons like webcam not open etc.
        continue
    
    cv2.imshow("Fewery",frame)
    cv2.imshow("Fewery_gray",gray_frame)
    #plotting both BGR(original) and grayscale frames
    
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
cap.release()
cv2.destroyAllwindows()