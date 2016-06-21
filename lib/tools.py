# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:06:17 2016

@author: David
"""

import numpy


def create_windowed_roi(input,startx,starty,width,height):
    """From a 2D array (input) containing brightness information of the original image, create a 2D array starting at the specified coordinates with the specified thicknesses, convoluted with a window function. Then, return the 2D array, normalized by having each element be the prior element's residual.
    The window function will be, by default, a Hamming function. (TODO: Look into other window functions.)"""

    roi = numpy.ndarray([width,height])
    
    winx = numpy.hamming(width)
    winy = numpy.hamming(height)
    
    mean = 0.0 #to be subtracted off each element

    
    for i in range(width):
        for j in range(height):
            roi[i][j] = winx[i]*winy[j]*input[startx+i][starty+j] #create windowed value
            mean += roi[i][j] #tally
    
    mean /= (i*j)
    
    for i in range(width):
        for j in range(height):
            roi[i][j] -= mean #replace original value with its residual, normalizing the array
    
    return roi
    

def calculate_relative_intensities(input, n_slices):
    """Input a n-dimentionsal array, and the number of subregions to be made (in a list of ints, n_slices; e.g., [2,3] would divide the original image into 6 rectangles, 2 fitting in the x-direction and 3 fitting in the y-direction). Return an n-dimensional array with the mean intensities of each subregion, relative to the mean intensity of the entire image.
    Will complain if the dimension of input does not match the size of n_slices."""
    
    if(input.ndim != n_slices):
        print('The dimension of the input data array and the specified slice numbers do not match!')
        pass
    
    intensities = numpy.ndarray(n_slices) #creates container for subsequent calculations
    
    roi_sizes = input.shape/n_slices    
    
    roi = numpy.ndarray(roi_sizes) #gives the roi its proper shape
    #TODO: check for rounding problems (since dividing ints by ints)
    
    total_sum = 0.0
    temp_sum = 0.0    
    
    #TODO: how to generalize looping properly?
    for i in range(n_slices[0]):
        for j in range(n_slices[1]): #for each roi
            roi = input[roi_sizes[0]*i:roi_sizes[0]*(i+1)][roi_sizes[1]*j:roi_sizes[1]*(j+1)]        #pull out the information on a given roi
        
            for a in range(roi_sizes[0]): #at each x-pixel within an roi
                for b in range(roi_sizes[1]): #at each y-pixel within an roi
                    temp_sum += roi[a][b]
            
            total_sum += temp_sum
            
            intensities[i][j] = temp_sum / roi.size() #mean intensity
            
            temp_sum = 0
    
    
    mean_intensity = total_sum / input.size()
    
    intensities /= mean_intensity #normalize intensities matrix by mean_intensity of entire image
    
    return intensities
    
    
    