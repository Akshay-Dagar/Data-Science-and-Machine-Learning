import cv2

img=cv2.imread("fury.png",cv2.IMREAD_GRAYSCALE)

cv2.imshow("Ours is the Fury",img)        #the string will be the title of the image

cv2.waitKey(0)                             
cv2.destroyAllWindows() 

#specifies how many milliseconds the image window should
#be open for (0 means infinitely open, until u close it)               
