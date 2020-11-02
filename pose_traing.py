from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam ,SGD
from keras.callbacks import ModelCheckpoint
import keras_metrics

import numpy as np


import pandas
x_data=np.load('x_data.npy')
y_data=np.load('y_data.npy')

len_x_data=len(x_data)
model = Sequential()
model.add(Dense(len_x_data, activation='relu', input_shape=(75,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))


model.add(Dense(y_data.shape[1], activation='softmax'))
model.compile(optimizer=Adam(0.005),
              loss='categorical_crossentropy',
              #loss='binary_crossentropy',
              metrics=['accuracy',keras_metrics.precision(),keras_metrics.recall()])
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]
model.fit(x_data, y_data, validation_split=0.33, epochs=200,batch_size=25,callbacks=callbacks_list)


#model.save('ThreeTypeAdam1st.h5')