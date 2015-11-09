# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:53:28 2015

@author: Anders Nymark Christensen 
based on Matlab code by Anders Nymark Christensen & William Sebastian Henning Benzon 2014
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.measure import regionprops

from scipy import misc



def BlobFunct(im, plotopt):
    """
    The BlobFunct is the main function in the blobdetection feature created
    at Rigshospitalet during Julie Bjerglund Andersen's project regarding
    bloodflow estimation in piglets using red, green and blue microspheres.
    
    In this script several parameters used for microsphere detection can be
    tweaked and modified to various situations.
    Some of the adjustable parameters:
    minBlobSize - Minimum Blobsize
    maxBlobSize - Maximum Blobsize
    edgeSigma - Used in edge detections
    shapeThreshold - Roundness measure
    channelThreshold - Minimum RGB values
    
    INPUT:
    im = RGB image (uint)
    plotopt = [0,1] if visuals during the script is needed, set plotopt = 1
    otherwise, plotopt = 0.
    
    OUTPUT:
    amount - total amount of microspheres detected [Red Green Blue]
    mask - Binary tissue mask
    arealsize = size of tissue (pixels)
    red - Coordinates for all red detected microspheres ([X Y])
    green - Coordinates for all green detected microspheres ([X Y])
    blue - Coordinates for all blue detected microspheres ([X Y])
    
    SUBFUNCTIONS:
    For this function to work, following functions are needed:
    areal_est
    RGBcheck
    Image Processing Toolbox(By MathWorks)
    
    
    NOTE:
    This function calls RGBcheck which also affects the outcome very
    drastically. So if adjusting parameters in this script doesnt affect the
    outcome. Please change settings in RGBcheck.m
    """
    # Settings
    nLoops = 1
    minBlobSize = 35.0
    maxBlobSize = 80.0
    edgeSigma = 3.0 # Used in the combined edge and blobdetection
    shapeThreshold = 2.0 # The maximun deviation from circular defined by abs(perimeter/diameter - pi) < shapeThreshold
    channelThreshold = [0, 0, 0] #[50 90 60];


    im = np.array(im)
    IM = np.copy(im)
    IM = IM.astype(np.float64, copy=False)
    
    mask, arealsize = areal_est(IM)
    misc.imsave('mask.png', mask)
    
    for index in range(0,3):
        if index == 1:
            maxBlobSize = 150
        else:
            maxBlobSize = 80
        
 
        # Find edges
        BW = edgeLoG(IM[:,:,index],[],edgeSigma)
        # The blobs are closed contours. We find them by filling the holes and
        # subtracting the original edges, leaving only closed contours in the
        # image.
        BW2 = scipy.ndimage.morphology.binary_fill_holes(BW) - BW
        
        
        # We remove areas smaller and larger than what we have specified
        IMsmall = morphology.remove_small_objects(BW2, min_size=minBlobSize, connectivity=8)
        IMbig = morphology.remove_small_objects(BW2, min_size=maxBlobSize, connectivity=8)
        IMblobs = (IMsmall - IMbig) > 0
        IMtemp, num = scipy.ndimage.label(IMblobs)
            
        
        # Label the remaining areas, if below the channel threshold they are
        # removed
        for nBlob in range(0,num):
            if np.max(IM[IMtemp==nBlob]) < channelThreshold[nLoops]:
                IMtemp[IMtemp==nBlob] = 0
  
        IMtemp2, num = scipy.ndimage.label(IMtemp)
        
        

        # Find centroid for plotting
        props = regionprops(IMtemp2) #'Centroid','Area','Perimeter','MinorAxisLength');
        
        allProps = np.zeros( (num, 2) )
        n = 0        
        for nProps in range(0,num):        
            allProps[n,:] = np.asarray(props[nProps].centroid)
            n += 1
        if index == 0:
            
            newProps = np.zeros( (num, 2) )
            # remove points that are not circular
            n = 0 
            for nProps in range(0,num):
                if abs(props[nProps].perimeter/props[nProps].minor_axis_length - np.pi) < shapeThreshold:                   
                    newProps[n,:] = np.asarray(props[nProps].centroid)
                    n += 1
            newProps = newProps[~np.all(newProps == 0,axis = 1), :]
            allProps = newProps
            
        
        xy1 = np.round(allProps)
        
        deletionList = []
        for i in range(0,np.size(xy1,0)):
            if mask[xy1[i,0],xy1[i,1]]==0:
                deletionList.append(i)


        xy1 = np.delete(xy1,deletionList,0)
        ## Blob-check
        if index == 0:
            red = np.copy(xy1)
            red = RGBcheck(im,red,'red')
        elif index ==1:
            green = np.copy(xy1)
            green = RGBcheck(im,green,'green')
        elif index ==2:
            blue = np.copy(xy1)
            blue = RGBcheck(im,blue,'blue')
  
    
    nLoops = nLoops +1
    amount = [len(red), len(green), len(blue)]
    
    if plotopt==1:
        plt.figure
        plt.imshow(im)
        plt.plot(red[:,1],red[:,0],'rx')
        plt.plot(green[:,1],green[:,0],'gx')
        plt.plot(blue[:,1],blue[:,0],'bx')
        plt.show()
        print('Plotting...')
    
    return(amount,mask,arealsize,red,green,blue)
    
    
    
    
def areal_est(im):
    """    
    Uses image to create binary image-mask of tissue corresponding to value 1
    and non-tissue = value zero. 
    INPUT:  im = RGB-image 
    OUTPUT: BW = Binary mask image
            arealsize = amount of pixels in BW = 1
    """
    from skimage.exposure import rescale_intensity
    from skimage.filter import threshold_otsu
    from skimage.morphology import remove_small_objects
    from skimage.morphology import disk
    from skimage.morphology import binary_closing

    minSlideSize = 200000
    
    imScaled = rescale_intensity(im[:,:,1], in_range=(np.mean(im[:,:,1])-9*np.std(im[:,:,1]), np.mean(im[:,:,1])+9*np.std(im[:,:,1])))
    # Calculating optimal binary threshold level
    level = threshold_otsu(imScaled)
    
    # Initial binary mask
    mask = imScaled > level
    
    # removing smal segments
    mask = remove_small_objects(mask, min_size=minSlideSize, connectivity=8)
    se = disk(10)
    mask = binary_closing(mask,se)
    
    # Isolating largest binary piece of tissue
    L, num = scipy.ndimage.label(mask);
    props = regionprops(L)
    
    largest = 0
    for index in range(0,num):
        if props[index].area > largest:
            largest = props[index].area
            largestIndex = index
    
    # Isolating primary label for binary image
    BW = (L == largestIndex)
    # Securing tissue-value = 1
    if BW[0,0] == True:
        BW = ~BW
        
    arealsize = largest;
    
    return(BW, arealsize)
    
    
    
def RGBcheck(im, blobs, color):
    """
    RGBcheck 
    RGBcheck tests wheter each blob fulfill color-composition requirements in
    RGB-components. Blobs must be rounded coordinates!
    INPUT:
    im:       Must be RGB image
    blobs:    2-dimensional coordinates with index not exceeding image size 
              and rounded to nearest whole-number
    color:    Must be a string, either "red", "green" or "blue".
    OUTPUT:
    blobs: Same vector as "blobs", but blobs which did not meet
              requirements have been removed.
              
    Red
    If the center blob-pixel has higer intensity in green or blue, the red
    blob is deleted. OR if the pixel intensity is below Rthresh1(80). To
    ensure homogenity in blob. Standard deviation must not be above Rthresh2
    (10) in 3by3 matrix with centroid as center pixel. 
    
    Green
    The blob is deleted if center pixel has higer intensity in red, if the
    green intensity timed Gthresh1 (1.25) is lower than blue intensity and if
    green intensity timed Gthresh2 (0.75) is higher than red intensity
    Further more blob is discarded if mean green blob intensity in 3by3
    matrix is less than Gthresh3 (85). 
    
    Blue
    If the center pixel intensity is higher in red og green channel, blob is
    discarded. If the center pixel does not have minimum Bthresh[2] times mean
    blue intensity in image, blob is discarded. IF the sum of red and green
    intensity is higher than blue intensity in center-pixel, blob is
    discarded. Last if 3by3 matrix with center equal to blob-center does not
    have a minimum blue intensity of Bthresh2 (130). Blob is discarded.          
    """
    # Threshold values:
    Rthresh1 = 0;
    Rthresh2 = 19;
    
    Gthresh1 = 1.15; # 1.25
    Gthresh2 = 0.55;
    Gthresh3 = 80;
    
    Bthresh1 = 1;
    Bthresh2 = 110; # 125
    
    a =  np.size(blobs,0)
    
    deletionarray = []
    # RED
    # Remove if green is more intense
    if color == 'red':
        for i in range(0,a):
            RGBs = im[blobs[i,0],blobs[i,1],:]
            if RGBs[1] > RGBs[0]:
                deletionarray.append(i)
            elif RGBs[2] > RGBs[0]:
                deletionarray.append(i)
            elif RGBs[0] < Rthresh1:
                deletionarray.append(i)
        
        blobs = np.delete(blobs,deletionarray,0)
        deletionarray = []
        a =  np.size(blobs,0)
        for i in range(0,a):
            if blobs.any:
                areal = sections_area(blobs[i,:],im,1)                
                if np.mean(areal[:,:,0]) < Rthresh1:
                    deletionarray.append(i)

        blobs = np.delete(blobs,deletionarray,0)
        deletionarray = []
        a =  np.size(blobs,0)
        for i in range(0,a):
            if blobs.any:
                areal = sections_area(blobs[i,:],im,1)
                RedVect = areal[:,:,0]
                if np.std( RedVect.astype(np.float64, copy=False) ) > Rthresh2:
                    deletionarray.append(i)

        blobs = np.delete(blobs,deletionarray,0)
        
        
        # GREEN
        # Remove if red is more intense OR if blue is more than 1.25x intense
        # Remove green if average green value in blob matrix (9x9) is less than 100
    elif color == 'green':
        for i in range(0,a):
            RGBs = im[blobs[i,0],blobs[i,1],:]
            if RGBs[0] > RGBs[1] or RGBs[2] > Gthresh1*RGBs[1] or RGBs[0] < Gthresh2*RGBs[1]:
                deletionarray.append(i) 
            
            
        a =  np.size(blobs,0)
        for i in range(0,a):
            if blobs.any:
                areal = sections_area(blobs[i,:],im,1)
                if np.mean(areal[:,:,1]) < Gthresh3:
                    deletionarray.append(i)

        blobs = np.delete(blobs,deletionarray,0)        
        # BLUE
        # If red is more intense OR green is more intense
        # If blue is less than 2 times the average blue
        # If blue is less than the sum of green and red
    elif color == 'blue':
        for i in range(0,a):
            deleted = 0;
            RGBs = im[blobs[i,0],blobs[i,1],:]
            if RGBs[1] > RGBs[2] or RGBs[0] > RGBs[2]:
                deletionarray.append(i)
                deleted = 1

            if deleted != 1:
                if RGBs[2] < Bthresh1*np.mean(im[:,:,2]):
                    deletionarray.append(i)
                    deleted = 1

            if deleted != 1:
                if RGBs[2] < sum(RGBs[0:1]):
                    deletionarray.append(i)
                    deleted = 1

        a =  np.size(blobs,0)
        for i in range(0,a):
            if blobs.any:
                areal = sections_area(blobs[i,:],im,1)
                if np.mean(areal[:,:,2]) < Bthresh2:
                    deletionarray.append(i)
                    
        blobs = np.delete(blobs,deletionarray,0)

    
    return(blobs)
    
    
    
def sections_area(blobs,im,sidelength):
    """
    sections
    Used for extracting features for a specific area given a size,
    coordinates and image. output-features are mean values and standard
    deviation.
    INPUT:
    blobs:      must be an x times 2 matrix with koordinates
    image:      must be a A x B x 3, uint8 image
    sidelength: adviced to be somewhere between 1 and 20
    Function is taking image-border coordinates into account and crops the
    sections evaluated to fit image dimension without resulting in error.
    By William Sebastian Henning Benzon 2014
    """
    rB = np.round(blobs)

    g = np.copy(im)
#    a = np.size(rB,0)
    x = np.size(g,0)
    y = np.size(g,1)

#    for i in range(0,1):
    if rB[1]-sidelength < 1:
        if rB[0]-sidelength < 1:
            section = g[1:rB[1]+sidelength,1:rB[0]+sidelength,:]
        elif rB[0]+sidelength > y:
            section = g[1:rB[1]+sidelength,rB[0]-sidelength:y,:]
        else:
            section = g[1:rB[1]+sidelength,rB[0]-sidelength:rB[0]+sidelength,:]
                
    elif rB[1]+sidelength > x:
        if rB[0]-sidelength < 1:
            section = g[rB[1]-sidelength:x,1:rB[0]+sidelength,:]
        elif rB[0]+sidelength > y:
            section = g[rB[1]-sidelength:x,rB[0]-sidelength:y,:]
        else:
            section = g[rB[1]-sidelength:x,rB[0]-sidelength:rB[0]+sidelength,:]
              
    elif rB[0]-sidelength < 1:
        section = g[rB[1]-sidelength:rB[1]+sidelength,1:rB[0]+sidelength,:]
    elif rB[0]+sidelength > y:
        section = g[rB[1]-sidelength:rB[1]+sidelength,rB[0]-sidelength:y,:]
    else:
        # Not border coordinates
        section = g[rB[1]-sidelength:rB[1]+sidelength,rB[0]-sidelength:rB[0]+sidelength,:]

    
    
    return(section)    
    
    
    
def edgeLoG(im, thres, edgeSigma):
    """
    Detect edges using Laplacian of Gassian
    INPUT:
    im: single channel image, type=float32/64
    thres: threshold for edges
    edgeSigma: Kernel used in Laplacian of Gaussian
    OUTPUT:
    fc: Binary image with edges
    Anders Nymark Christensen, 2015
    """
    import scipy.ndimage as nd
            
    if not edgeSigma:
        edgeSigma = 2.0

    LoG = nd.gaussian_laplace(im, edgeSigma)         
        
    if not thres:
        thres = np.absolute(LoG).mean() * 0.75

    #misc.imsave('LoG.png',LoG)

    # Find zero-crossing
    zc1 = (np.diff(np.sign(LoG),1,0) == -2) & (np.abs(np.diff(LoG,1,0)) > thres )
    zc2 = (np.diff(np.sign(LoG),1,0) == 2) & (np.abs(np.diff(LoG,1,0)) > thres )
    zc12 = np.pad(np.logical_or(zc1,zc2),((0,1),(0,0)),'constant', constant_values=((False,False),(False,False))) 
    
    zc3 = (np.diff(np.sign(LoG),1,1) == -2) & (np.abs(np.diff(LoG,1,1)) > thres ) 
    zc4 = (np.diff(np.sign(LoG),1,1) == 2) & (np.abs(np.diff(LoG,1,1)) > thres ) 
    zc34 = np.pad(np.logical_or(zc3,zc4),((0,0),(0,1)),'constant', constant_values=((False,False),(False,False)))    
    
    zc = np.logical_or(zc12,zc34)
    zc = np.logical_or(zc, LoG == 0)
    
    return zc