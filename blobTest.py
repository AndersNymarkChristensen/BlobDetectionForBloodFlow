# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:54:17 2015

@author: anym
"""

from scipy import misc
im = misc.imread('C:\Projektor\BlobDetectionForBloodFlow\IMG_0039.JPG')

import blobFunct as bf

plotopt = 1
amount,mask,arealsize,red,green,blue = bf.BlobFunct(im, 1)