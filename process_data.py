import numpy as np
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical


stand=np.load('stand.npy')
sit=np.load('sit.npy')
test=np.load('test.npy')


#make label and merge label
labels = np.zeros(len(stand))
labels = np.append(labels, np.full((len(sit)), 1))
labels = np.append(labels, np.full((len(test)), 2))

#merge data
dataset=np.vstack((stand,sit))
dataset=np.vstack((dataset,test))

#shuffle all data
x_data, y_data = shuffle(dataset, labels)


#one-hot enconding
y_data= to_categorical(y_data, 3)  # we have 3 categories, dab, tpose, other

#resize x_data
x_data[:,:,0] = x_data[:,:,0] / 720 # I think the dimensions are 1280 x 720 ?
x_data[:,:,1] = x_data[:,:,1] / 1280  # let's see?

#remove confidence dim
#x_data = x_data[:,:,:2]

#flat X_data
x_data = x_data.reshape(len(x_data), 75)      # we got rid of confidence percentage

np.save('x_data.npy',x_data)
np.save('y_data.npy',y_data)

print('complete to process data')
