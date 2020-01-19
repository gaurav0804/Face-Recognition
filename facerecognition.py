import cv2
import sys
import os
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from keras.models import load_model

path='./images'

model=load_model('facenet_keras.h5')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

images=[]
labels=os.listdir(path)
i=0

for label in labels:
    folderpath=path+'/'+label
    
    anchor_list=os.listdir(folderpath)
    
    for anchor in anchor_list:
        
        img=cv2.imread(folderpath+'/'+anchor)
        img=cv2.resize(img,(160,160))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        images.append(img)

images=np.array(images)
labels=np.array(labels)
embeds=model.predict(images)
img_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    images=[]
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        crop_img=frame[x:x+w,y:y+h]
        crop_img=cv2.resize(crop_img,(160,160))
        crop_img=cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
        crop_img = crop_img.astype('float32')
        mean, std = crop_img.mean(), crop_img.std()
        crop_img = (crop_img - mean) / std
        crop_img=crop_img.reshape(1,160,160,3)
        y_hat=model.predict(crop_img)
        label='unknown'
        for i in range(len(embeds)):
            embed=embeds[i]
            embed=embed.reshape(1,embed.shape[0])
            print(embed.shape)
            print(y_hat.shape)
            dist=pairwise_distances(embed,y_hat)
            print(str(labels[i])+' : '+str(dist))
            
            if dist<5:
                label=labels[i]
        cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)
        
    cv2.imshow('FaceDetection', frame)
    # Display the resulting frame
        
    
    

    if k%256 == 27: #ESC Pressed
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "facedetect_webcam_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
