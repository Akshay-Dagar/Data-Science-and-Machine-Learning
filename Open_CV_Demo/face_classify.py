import cv2
import numpy as np
import os
from collections import Counter

########################################################### Fetching Training Data #########################################################

train_data=[]                                                          #will store the feature values for each image
train_labels=[]                                                        #the labels of the training images (0,1,2,3 etc.)
names={}                                       #will store the names to be displayed corresponding to the predicted class (0,1,2,3 etc.)

class_label=0

for fname in os.listdir("train_data"):                                #returns a list containing names of all files inside train_data folder
    if fname[-4:]==".npy":                                             #we only want .npy files' data
        data=np.load("train_data/"+fname)                              #loading each file, data is a numpy array now
    
        train_data.append(data)
        train_labels.append([class_label]*data.shape[0])              
        #since data file has data of multiple images, we will create a list whose each element correponds to id of each image in the file
        #and since in a partiular file, the ids of all images will be the same (as each file corresponds to one person only), we are keeping 
        #the class_label same for all images of a particular file.

        names[class_label]=fname[:-4]                                  #removing the ".npy" at the end
        class_label+=1                                                 #for the next file (next person's images)
        
train_data=np.concatenate(train_data,axis=0)
train_labels=np.concatenate(train_labels,axis=0).reshape(-1,1)         #we want it in 1 column format
train=np.concatenate((train_data,train_labels),axis=1)

print("Training Data fetched from: ./train_data folder")
print("Training Data Shape:",train.shape)


########################################################## KNN Function ####################################################################
def distance(x1,x2,p=2):
    if p==1:                                           #L1 distance (Manhattan)
        return abs(x1-x2)
    else:                                              #L2 distance (Euclidean)
        return (sum((x1-x2)**2))**0.5
    
def KNN(X,Y,x,k=5,p=2):                                #(X=train_data,Y=train_labels,x=test_data)
    
    dist=[]                                            #will store distance of x 
    for i in range(X.shape[0]):
        d=distance(X[i],x,p)                           #calculating distance between x and eachpoint in training data (X[i])
        dist.append((Y[i],d))
    
    dist=sorted(dist,key=lambda x:x[1])                #sorting in ascedning order of distances
    dist=dist[:k]                                      #we choose the K nearest points only
    
    kLabels=[]
    for d in dist:
        kLabels.append(d[0])
        
    c=Counter(kLabels)
    
    return c.most_common(1)[0][0]                      #return the label with highest frequecy as predicted value for x

########################################################## Classifying: ####################################################################

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)                                        #intialize camera
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")        #intializing cv2 module's CascadeCalssifer class's object

while 1:
    
    ret,frame=cap.read()                                                     
    
    if ret==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        
        offset=0
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))                      
        #we have the test image, but this is 3D (100*100*3, (3 is for channels), 
        #we need to convert to 1D before passing it to KNN and predicting
        face_section=face_section.flatten()
        prediction=KNN(train[:,:-1],train[:,-1],face_section,7)
        
        predicted_name=names[prediction]        #prediction is a no., we need to get the corresponding name using names dict
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,100),5)
        cv2.putText(frame,predicted_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(200,50,50),2,cv2.LINE_AA)
        
    cv2.imshow("Fewery",frame)
    
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
    