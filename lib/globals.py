# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:13:12 2016

@author: David Khachatrian
"""

import numpy as np #for fast Fourier Transform
from matplotlib import pyplot
from PIL import Image



scale_min_dict = {'ori': None, 'c': None, 'o': None, 'e': 0, 'r': 0} #what to map the minimum value of the data to, for the purposes of converting HSV->RGB. 'None' implies the minimum value of the data will not necessarily rescale to a particular value -- the max value will determine the scaling factor


def init():
    pass