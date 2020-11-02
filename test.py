import keras
import keras_metrics

import numpy as np

params = dict()
params["model_folder"] = "../../../models/"

x_data=np.load('x_data.npy')
mpose = keras.models.load_model('weights-improvement-182-0.98.hdf5',custom_objects={'binary_precision':keras_metrics.precision(), 'binary_recall':keras_metrics.recall()})

for i in x_data:
    i = i.reshape(1, 75)
    output = mpose.predict_classes(i)

    print(output)