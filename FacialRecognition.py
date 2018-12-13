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
seed = 7
np.random.seed(seed)


emotions = ["fear","contempt","disgust","neutral", "anger", "sadness", "happy", "surprise"]


def get_files(emotion,check): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("%s\\%s\\*" %(check,emotion))
    return files

def make_sets(check):
    X_train=[]
    y_train=[]
    for i,emotion in enumerate(emotions,start=0):
        files=get_files(emotion,check)
        for file in files:
            img=cv2.imread(file)
            img=cv2.resize(img,(28,28))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X_train.append(img)
            y_train.append(i)
        print(len(X_train),len(y_train))
        
    return X_train,y_train
            

X_train,Y_train=make_sets('train')
x_test,y_test=make_sets('test')

X_train=np.array(X_train)
Y_train=np.array(Y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
# load data
#(X_train, Y_train), (x_test, y_test) = mnist.load_data()
print(Y_train[0])


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
x_test = x_test / 255

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, Y_train, validation_data=(x_test, y_test), epochs=100, batch_size=20, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#    print(np.argmax(model.predict(x_test[4].reshape(1,1,28,28))))
  #  plt.imshow(x_test[4].reshape(28,28))

model.save('facial_recognition_cnn.hd5')