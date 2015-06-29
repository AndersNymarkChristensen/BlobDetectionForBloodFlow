# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:54:17 2015

@author: anym
"""

from scipy import misc
im = misc.imread('C:/Dropbox/NeoFlow Artikel/Kode/PythonCode/IMG_0039.jpg')

import blobFunct as bf

amount,mask,arealsize,red,green,blue = bf.BlobFunct(im, 1)