import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
K.set_image_dim_ordering('th')
import glob
import cv2

# fix random seed for reproducibility


out=[]
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=load_model('facial_recognition_cnn_1.hd5')

emotions = ["neutral", "anger", "sadness", "happy", "surprise"]
def detect(grey,frame):
    faces=face_cascade.detectMultiScale(grey,1.3,5)
    for(x,y,w,h) in faces:
        roi_grey = grey[y:y+h, x:x+w]
        out = cv2.resize(roi_grey, (28,28))
        out=np.array(out)
        out=out.reshape(1,28,28)
        pred = model.predict(out.reshape(1,1,28,28))  
        pred=np.argmax(pred)
        return pred

def show(grey,frame,no):
    faces=face_cascade.detectMultiScale(grey,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        if(no==1):
            name="neutral"
        elif(no==2):
            name="anger"
        elif(no==3):
            name="sadness"
        elif(no==4):
            name="happy"
        else:
            name="surprise"
        print(name)
        color=(0,0,255)
        stroke=2
        cv2.putText(frame,name,(x,y),font,1,color,stroke)
        
    return frame



"""img = cv2.imread('24.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
i=0
while(i<2):
    img=show(gray,img,no)
    no=detect(gray,img)
    print(emotions[no])
    i=i+1"""
video_capture=cv2.VideoCapture(0)
while True:
    _,frame=video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=show(gray,frame,no)
    no=detect(gray,frame)
    if(no==1):
            name="neutral"
    elif(no==2):
            name="anger"
    elif(no==3):
            name="sadness"
    elif(no==4):
            name="happy"
    else:
            name="surprise"
    cv2.imshow('video',canvas)
    if cv2.waitKey(1) & 0XFF ==ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
"""cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""