# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:04:08 2016

@author: David
"""


#os.path.join(path, path, ...)
#### To ensure the working directory starts at where the script is located...
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
dep = os.path.join(dname, 'dependencies')
os.chdir(dname)
####



import numpy #for fast Fourier Transform
from PIL import Image
from lib import tools as t


outdir = os.path.join(dname, 'outputs') #directory for output files

#Take in image file, as bytearray

im_orig = Image.open(os.path.join(dep, 'test-orientation.tif'))
im = im_orig.convert('L') #convvert to grayscale

xsize, ysize = im.size

# in grayscale: 0 = black, 255 = white
data = numpy.array(im)


numx = 20
numy = 20

roix = int(xsize / numx) #roix = xsize for a given region of interest
roiy = int(ysize / numy)

#calculate unaltered mean intensities of the roi's, and the mean intensity of the entire image, to weight the anisotropy eigenvalue-ratio index later on

intensities = t.calculate_relative_intensities(input=data, slice_numbers=(numx, numy))

A_er = numpy.ndarray(intensities.shape)

for i in range(0, xsize, roix):
    for j in range(0, ysize, roiy):
        roi = t.create_windowed_roi(input=data, startx=i, starty=j, width=roix, height=roiy) #take subsections, have them normalized
        roi_f = numpy.fft.fftn(roi, s=None, axes=None, norm=None) #perform discrete Fourier transform on ROI
        #could potentially do a real FFT/Hermitian FFT. Could save computational time?
        
        # create radial histogram/profile ## TODO: Why? How?
        
        cov = numpy.dot(numpy.transpose(roi_f), roi_f) / roiy #obtain the covariance matrix, an roix-by-roix matrix
        
        #evals,evecs = numpy.linalg.eig(cov)
        
        evals_arr, evecs_arr = numpy.linalg.eig(cov)
        estuff = list(zip(evals_arr, evecs_arr))  #build up list of tuples of eignevalues and eigenvectors
        
#        for i in range(len(evals_arr)): #len(evals_arr) == len(evecs_arr), always
#            estuff.append(tuple(zip(evals_arr[i],evecs_arr[i]))) #build up list of tuples of eignevalues and eigenvectors
        
        estuff = sorted(estuff, key= lambda tup: tup[0]) #sort eigenvalues in ascending order, moving eigenvectors along with them
        
        A_er[i][j] = (1 - abs(estuff[0][0]/estuff[-1][0])) * intensities[i][j] #populate A_er matrix using appropriate formula and weighting
        #eigenvalues should already be working
        

with open(os.path.join(outdir, 'aniso_ratios.txt'), 'w') as outf:
    outf.write(A_er) #write resulting intensity array to file




#Perform anisotropy analysis at every pixel of the image
# im_d = Image.eval(im, anisotropy_analyze)

