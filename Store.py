
import tkinter.messagebox
import tkinter as tk
import numpy as np
import os

class Store():


    def SelectAndLabel (datum,label):
        root = tk.Tk()
        root.withdraw()

        if datum.poseKeypoints.any() and datum.poseKeypoints.ndim == 3:
            # if tk.messagebox.askokcancel(message='Do you want to save ?') == True:
            #
            #     for i in range(datum.poseKeypoints.shape[0]):
            #         label.append(datum.poseKeypoints[i])
            #     print('successfully save')
            # else:
            #     print('cancel')
            #
            # return label
            for i in range(datum.poseKeypoints.shape[0]):
               label.append(datum.poseKeypoints[i])
               return label
        else:

            print("nobody")
            return label

    def SaveLabelData(data,label_name):

        if data !=[] :

            data=np.asarray(data)
            if os.path.isfile(label_name+".npy") :
                old_data=np.load(label_name+".npy")
                data=np.vstack((old_data,data))
                print(data.shape)
                np.save(label_name+'.npy', data)
            else:
                print(data.shape)
                np.save(label_name+'.npy', data)
        else:
            print(label_name+' is empty')


