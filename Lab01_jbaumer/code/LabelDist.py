# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:52:09 2021

@author: Jor Fergus Dal
"""

import numpy as np
import h5py
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()

CM = np.zeros((3,3))

CM[0,:] = np.array([2.57476e+08, 1.89577e+06, 2.3189e+07])
CM[1,:] = np.array([3.27357e+07, 274487, 1.75019e+06])
CM[2,:] = np.array([115424, 22, 2.94942e+07])

n_BG = sum(CM[0,:])
n_PO = sum(CM[1,:])
n_CL = sum(CM[2,:])

Tot = n_BG + n_PO + n_CL

perc_BG = n_BG/Tot
perc_PO = n_PO/Tot
perc_CL = n_CL/Tot

labels = "Background", "Palm Oil Tree", "Clouds"
sizes = [n_BG, n_PO, n_CL]
colors = 'tan', 'g', 'lightgrey'

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
fig1.set_label('Label percentages for the data used for training:')
plt.show()
