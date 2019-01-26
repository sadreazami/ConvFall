
from __future__ import print_function
 
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import keras 
from keras.callbacks import ReduceLROnPlateau
from keras.layers.core import Flatten, Dense, Dropout, Lambda
#from keras.layers import *
from keras import backend as K
import cv2

def global_average_pooling(x):
    return K.mean(x, axis = (2))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

p=Lambda(global_average_pooling, output_shape=global_average_pooling_shape)
    
X = sio.loadmat('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/ts.mat')
X=X['Data']
#XX = sio.loadmat('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/tsnoise.mat')
#XX=XX['Data3']
#XXX = sio.loadmat('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/tstestnoise.mat')
#XXX=XXX['Data33']
import csv
with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/lab.csv', 'r') as mf:
     re = csv.reader(mf,delimiter=',',quotechar='|')
     re=np.array(list(re))
     label = re.astype(np.float64)
     Y_t=np.squeeze(label) 
  
nb_epochs = 3

y_train =Y_t[:158]
#y_train=np.tile(y_train,10) 
y_test =Y_t[158:]
#y_test=np.tile(y_test,10) 
   
x_train=X[:158]
#x_train=XX
x_test=X[158:]
#x_test=XXX

nb_classes = len(np.unique(y_test))
batch_size = min(x_train.shape[0]/8, 16)

y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std) 
x_test = (x_test - x_train_mean)/(x_train_std)

x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))
#x_train = np.transpose(x_train, (0, 2, 1))
#x_test = np.transpose(x_test, (0, 2, 1))

input_shape=x_train.shape[1:]

x = keras.layers.Input(x_train.shape[1:])
#    drop_out = Dropout(0.2)(x)
conv1 = keras.layers.Convolution1D(300, 9, padding='same')(x)
conv1 = keras.layers.normalization.BatchNormalization()(conv1)
conv1 = keras.layers.Activation('relu')(conv1)

conv2 = keras.layers.Convolution1D(200, 5, padding='same')(conv1)
conv2 = keras.layers.normalization.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

conv3 = keras.layers.Convolution1D(100, 3, padding='same')(conv2)
conv3 = keras.layers.normalization.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

#conv4 = keras.layers.Convolution1D(8, 5, padding='same')(conv3)
#conv4 = keras.layers.normalization.BatchNormalization()(conv4)
#conv4 = keras.layers.Activation('relu')(conv4)
#conv4 = keras.layers.Reshape((,8,3000))(conv4)
#conv4 = conv4.K.transpose((0, 2, 1))
#conv4 = K.permute_dimensions(conv4,(0,2,1))

full = p(conv3)    
out = keras.layers.Dense(nb_classes, activation='softmax')(full)

model = Model(input=x, output=out)

 
optimizer = keras.optimizers.Adam() #'sgd'
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
 
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=500, min_lr=0.001) 
hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
          verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr])
predict = model.predict(x_test)
preds = np.argmax(predict, axis=1)

#Print the testing results which has the lowest training loss.
log = pd.DataFrame(hist.history)
print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

labels = {1:'Non-Fall', 2:'Fall'}
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(preds, y_test,
                            target_names=[l for l in labels.values()]))

conf_mat = confusion_matrix(preds, y_test)

fig = plt.figure(figsize=(2,2))

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
#cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(2), [l for l in labels.values()])

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

# Testing
test=x_test[2:3,:,:]
test = np.random.random(input_shape)[np.newaxis,:]
layer_outs = functor([test, 1.])

class_weights = model.layers[-1].get_weights()[0]
conv_outputs = np.squeeze(layer_outs[-2])

#Create the class activation map.
cam = np.zeros(dtype = np.float32, shape =3000)
w0=class_weights.T[0,:]
w1=class_weights.T[1,:]
#for i, w in enumerate(class_weights.T[0,:]):
cam = w0 * conv_outputs
cam /= np.max(cam)

x_test_mean = x_test.mean()
x_test_std = x_test.std()
x_test_max = x_test.max()
plt.figure()
x_test= (x_test)/(x_test_max)
plt.plot(x_test[0,:,0]) #normalize???
plt.plot( cam, color='g') # scaleback ????
plt.plot( w0/w0.max(), color='k')
plt.plot( w1/w1.max(), color='y')
plt.plot( conv_outputs/conv_outputs.max(), color='r')   

#cam = cv2.resize(cam, (height, width))
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
oo=heatmap[:,:,0]
plt.plot(oo/oo.max())
#heatmap[np.where(cam < 0.2)] = 0
#img = heatmap*0.5 + original_img
#cv2.imwrite(output_path, img)
