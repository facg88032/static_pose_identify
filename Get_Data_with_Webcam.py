import cv2
from openpose import pyopenpose as op
import time
from Store import *



# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"

poseModel = op.PoseModel.BODY_25
original_keypoints_index = op.getPoseBodyPartMapping(poseModel)
keypoints_index = dict((bp, num) for num, bp in original_keypoints_index.items())

#open webcam
vs = cv2.VideoCapture(0)

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Create objects to process pictures
datum = op.Datum()


stand = []
sit=[]
lie=[]
test=[]

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


    fps_time = time.time()


    cv2.imshow("Openpose", image)
    # quit with a q keypress, z,x.. to label different data
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("z"):
        stand=Store.SelectAndLabel(datum,stand)
        print('stand:',len(stand))
    elif key == ord("x"):
        sit = Store.SelectAndLabel(datum, sit)
        print('sit:', len(sit))
    elif key == ord("c"):
        lie = Store.SelectAndLabel(datum, lie)
        print('lie:', len(lie))
    elif key == ord("v"):
        test = Store.SelectAndLabel(datum, test)
        print('test:', len(test))
    print(datum.poseKeypoints)
# clean up after yourself
vs.release()
cv2.destroyAllWindows()

#Save numpy form of label data
Store.SaveLabelData(stand,label_name='stand')
Store.SaveLabelData(sit,label_name='sit')
Store.SaveLabelData(lie,label_name='lie')
Store.SaveLabelData(test,label_name='test')
