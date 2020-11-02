import numpy as np
import cv2
from openpose import pyopenpose as op


data=np.load('sit.npy')
print(data.shape)

ct_seq=[1,2,3,4,3,2,1,5,6,7,6,5,1,0,16,18,16,0,15,17,15,0,1,8,9,10,11,24,11,22,23,22,11,10,9,8
        ,12,13,14,21,14,19,20]

img = np.zeros((720,1280, 3), np.uint8)
img.fill(200)
No_img=1480
count=No_img

while True:


    cv2.imshow("catch", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("z"):
        if No_img < len(data):
            img = np.zeros((720, 1280, 3), np.uint8)
            img.fill(200)
            for i in range(len(ct_seq) - 1):
                if (data[No_img][ct_seq[i]][0]==0 and data[No_img][ct_seq[i]][1]==0) or (data[No_img][ct_seq[i + 1]][0] ==0 and data[No_img][ct_seq[i + 1]][1]==0):
                    pass
                else:
                    cv2.line(img, (data[No_img][ct_seq[i]][0], data[No_img][ct_seq[i]][1]),
                             (data[No_img][ct_seq[i + 1]][0], data[No_img][ct_seq[i + 1]][1]), (0, 0, 0), 5)
            count+=1
            cv2.putText(img,
                        str(count),
                        (500,500), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 0), 5)
            cv2.imshow("catch", img)

            No_img=No_img+1
        else:
            print("This is last picture")

cv2.destroyAllWindows()