import numpy as np
import cv2
data=np.load("sit.npy")

count=1
while True:
    try:
        index = int( input("input incomplete data index:"))
        try:
            if index != 0:
              data = np.delete(data, index-count,axis=0)
              count+=1
              print('sucess')
            else:
                break
        except IndexError as e:
            print(e)
    except ValueError:
        print("index of type require int ")



print(data.shape)
np.save("sit.npy",data)







