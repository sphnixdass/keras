import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import pandas as pd
from pandas import DataFrame


# The competition datafiles are in the directory ../input
# Read competition data files:
# Backend - Theano
train = pd.read_csv("input/train.csv")
test  = pd.read_csv("input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

np.random.seed(1337) 

batch_size = 128
nb_classes = 10
nb_epoch = 20

# convert class vectors to binary class matrices
X_train=train.iloc[:,1:].as_matrix().astype('float32')
X_test=test.as_matrix().astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(train['label'], nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer= RMSprop(),
              metrics=['accuracy'])

fit1 = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
pred=model.predict(X_test)
pred2=[]
for i in range(pred.shape[0]):
    pred2.append(list(pred[i]).index(max(pred[i])))
out_file = open("predictions.csv", "w")
out_file.write("ImageId,Label\n")
for i in range(len(pred2)):
    out_file.write(str(i+1) + "," + str(int(pred2[i])) + "\n")
out_file.close()