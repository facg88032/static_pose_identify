import cv2
from openpose import pyopenpose as op
import keras
import keras_metrics

import time
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"

mpose = keras.models.load_model('weights-improvement-109-0.98.hdf5',custom_objects={'binary_precision':keras_metrics.precision(), 'binary_recall':keras_metrics.recall()})

poseModel = op.PoseModel.BODY_25
original_keypoints_index = op.getPoseBodyPartMapping(poseModel)
keypoints_index = dict((bp, num) for num, bp in original_keypoints_index.items())

vs = cv2.VideoCapture(0)

# Starting OpenPoseasdasdas
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()

fps_time = 0


while True:
    ret_val, frame = vs.read()

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # need to be able to see what's going on
    image = datum.cvOutputData
    cv2.putText(image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1)




    if datum.poseKeypoints.any() and datum.poseKeypoints.ndim == 3:
        first_input = datum.poseKeypoints
        try:

            #
            # first_input = first_input[:, :, :2]
            first_input[:, :, 0] = first_input[:, :, 0] / 720
            first_input[:, :, 1] = first_input[:, :, 1] / 1280


            for i in first_input:
                x_up = 1
                y_up = 1

                for j in range(25):
                    if i[j][1]<y_up and i[j][1] !=0 :
                        # print(y_up)
                        y_up=i[j][1]
                        if i[j][0]<x_up and i[j][1] !=0 :
                            x_up=i[j][0]

                i = i.reshape(1, 75)
                output = mpose.predict_classes(i)
                for j in output:
                   # print('output', j)
                   if j == 0:
                       # print("stand")
                       cv2.putText(image,
                                   "stand",
                                   (int(x_up * 720) + 50, int(y_up * 1280)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                   (0, 0, 0), 5)
                   elif j == 1:
                       # print("sit")
                       cv2.putText(image,
                                   "sit",
                                   (int(x_up * 720) + 50, int(y_up * 1280)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                   (0, 0, 0), 5)
                   elif j == 2:
                       # print('other')
                       cv2.putText(image,
                                   "other",
                                   (int(x_up * 720) + 50, int(y_up * 1280)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                   (0, 0, 0), 5)







        except:
            continue

    fps_time = time.time()
    cv2.imshow("Openpose", image)
    # quit with a q keypress, b or m to save data
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        poseModel = op.PoseModel.BODY_25
        break


# clean up after yourself
vs.release()
cv2.destroyAllWindows()


