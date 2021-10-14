# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:45:59 2021

@author: Jor Fergus Dal
"""

import numpy as np
import h5py
import matplotlib.pylab as plt

np.seterr(divide='ignore', invalid='ignore')

def calc_NDVI(NIR, RED):
    NDVI = (NIR.astype(float)-RED.astype(float))/(NIR+RED)
    return NDVI

h5 = h5py.File("dataset_test.h5","r")

NIR = h5["NIR"]
RGB = h5["RGB"]
RED = RGB[:,:,:,0]

plt.imshow(RGB[0])
plt.show()
plt.imshow(RGB[0,:,:,0], cmap = plt.cm.Reds)
plt.show()

ndvi = calc_NDVI(NIR, RED)

plt.imshow(ndvi[0], cmap = plt.cm.summer)
plt.show()


