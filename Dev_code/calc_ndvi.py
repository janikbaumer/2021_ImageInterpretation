# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:14:49 2021

@author: Jor Fergus Dal
"""


def calc_NDVI(NIR, RED):
    """
    This function takes the NIR and RED channels of an image and returns the NDVI index.
    NIR and RED values are first normalized.
    Make sure to pass the parameters are matrices of exactly 2 dimensions.
    Also, important that zero divisions are handled properly.
    One solution, use following setting: np.seterr(divide='ignore', invalid='ignore')

    Parameters
    ----------
    NIR : nparray
        Array of 2 dim, where dims indicate pixels, and value represents NIR intensity.
    RED : nparray
        Array of 2 dim, where dims indicate pixels, and value represents RED intensity.

    Returns
    -------
    NDVI : nparray
        Array of 2 dim, where dims indicate pixels, and value represents NDVI, a value between -1 and 1.

    """
    
    NIR = NIR[:]
    NIR = NIR/((2**16)-1)
    RED = RED[:]
    RED = RED/((2**8)-1)
    
    NDVI = (NIR.astype(float)-RED.astype(float))/(NIR+RED)
    return NDVI
