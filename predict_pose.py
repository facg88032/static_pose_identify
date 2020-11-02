import keras
import numpy as np
modello = keras.models.load_model('dab-tpose-other.h5')
dabDataset = np.load('test.npy')
dabDataset[:,:,0] = dabDataset[:,:,0] / 720 # I think the dimensions are 1280 x 720 ?
dabDataset[:,:,1] = dabDataset[:,:,1] / 1280  # let's see?
dabDataset = dabDataset[:,:,1:]
dabDataset = dabDataset.reshape(len(dabDataset), 50)
output=modello.predict_classes(dabDataset) # returns array([1, 1, 1, 1, 1, 1])

for j in output:
    if j == 0:
        print('stand')
    elif j == 1:
        print('sit')
    else:
        print('other')