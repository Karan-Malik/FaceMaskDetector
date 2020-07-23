# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:56:47 2020

@author: Karan
"""


import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import adam
from keras.preprocessing import image
import cv2
import datetime


# UNCOMMENT THE FOLLOWING CODE TO TRAIN THE CNN FROM SCRATCH

# BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK
'''
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

model_saved=model.fit_generator(
        training_set,
        epochs=10,
        validation_data=test_set,

        )

model.save('mymodel.h5',model_saved)

#To test for individual images

#test_image=image.load_img('C:/Users/Karan/Desktop/ML Datasets/Face Mask Detection/Dataset/test/without_mask/30.jpg',target_size=(150,150,3))
test_image=image.load_img(r'C:\Users\Karan\Pictures\Camera Roll/21.jpg',
                          target_size=(150,150,3))
test_image
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
type(mymodel.predict_classes(test_image)[0][0])
'''

# IMPLEMENTING LIVE DETECTION OF FACE MASK

mymodel=load_model('mymodel.h5')

cap=cv2.VideoCapture(0)

while cap.isOpened():
    _,img=cap.read()
    face_img=cv2.resize(img,(150,150))
    face_img=face_img/255.0
    face_img=np.reshape(face_img,(150,150,3))
    face_img=np.expand_dims(face_img,axis=0)
    pred=mymodel.predict_classes(face_img)
    if pred[0][0]==1:
        cv2.putText(img,'NO MASK',(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    else:
        cv2.putText(img,'MASK',(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
    datet=str(datetime.datetime.now())
    cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        
    cv2.imshow('img',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
   
