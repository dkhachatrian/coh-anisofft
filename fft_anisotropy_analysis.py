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

import sys #to terminate early, if necessary

import numpy #for fast Fourier Transform
from PIL import Image
from lib import tools as t


outdir = os.path.join(dname, 'outputs') #directory for output files

#Take in image file, as bytearray

#image_name = input("Please state the full filename for the image of interest (located in the dependencies directory of this script), or enter nothing to quit: \n")
#
#while not os.path.isfile(os.path.join(dep, image_name)):
#    if image_name == '':
#        sys.exit()
#    image_name = input("File not found! Please check the spelling of the filename input. Re-enter filename (or enter no characters to quit): \n")
#
#im_orig = Image.open(os.path.join(dep, image_name))


im_orig = Image.open(os.path.join(dep, 'test-orientation.tif'))

im = im_orig.convert('L') #convvert to grayscale

(xsize, ysize) = im.size #im is still an Image, and not an array

# in grayscale: 0 = black, 255 = white
data = numpy.array(im)

# TODO: collapse below into a loop

inx = input("Please state the number of regions of interest you would like to fit in the x-direction of the image. There must be no remainder. (Current x-size of image: " + str(int(xsize)) + "): \n")

while (xsize % int(inx)) != 0:
    inx = input("Does not divide cleanly! Please try again. (Current x-size of image: " + str(int(xsize)) + "): \n")
    

iny = input("Please state the number of regions of interest you would like to fit in the y-direction of the image. There must be no remainder. (Current y-size of image: " + str(int(ysize)) + "): \n")

while (ysize % int(iny)) != 0:
    iny = input("Does not divide cleanly! Please try again. (Current y-size of image: " + str(int(ysize)) + "): \n")
    
numx = int(inx)
numy = int(iny)
#numx = 20
#numy = 20

roix = int(xsize / numx) #roix = xsize for a given region of interest
roiy = int(ysize / numy)

#calculate unaltered mean intensities of the roi's, and the mean intensity of the entire image, to weight the anisotropy eigenvalue-ratio index later on

intensities = t.calculate_relative_intensities(input=data, slice_numbers=(numx, numy))

A_er = numpy.ndarray(intensities.shape)

for i in range(0, numx):
    for j in range(0, numy):
        roi = t.create_windowed_roi(input=data, startx=i*roix, starty=j*roiy, width=roix, height=roiy) #take subsections, have them normalized
        roi_f = numpy.fft.fftn(roi, s=None, axes=None, norm=None) #perform discrete Fourier transform on ROI
        #could potentially do a real FFT/Hermitian FFT. Could save computational time?
        #TODO: Should we normalize? (i.e. make it 'ortho')
        
        # create radial histogram/profile ## TODO: Why? How?
        
        
        p_x = [(n.real)**2 for n in numpy.reshape(roi_f, roi_f.size)] #reshape() changes the matrix to a (in this case) 1D array
        p_y = [(n.imag)**2 for n in numpy.reshape(roi_f, roi_f.size)]
        
        P = numpy.transpose(numpy.array([p_x, p_y])) #something seems off... That I have to transpose
        
        cov = numpy.dot(numpy.transpose(P), P) / P.shape[1] #obtain the covariance matrix, an roix-by-roix matrix
        #but how to normalize? This is correct
        
        #evals,evecs = numpy.linalg.eig(cov)
        
        evals_arr, evecs_arr = numpy.linalg.eig(cov)
        estuff = list(zip(evals_arr, evecs_arr))  #build up list of tuples of eignevalues and eigenvectors
        
#        for i in range(len(evals_arr)): #len(evals_arr) == len(evecs_arr), always
#            estuff.append(tuple(zip(evals_arr[i],evecs_arr[i]))) #build up list of tuples of eignevalues and eigenvectors
        
        estuff = sorted(estuff, key= lambda tup: tup[0]) #sort eigenvalues in ascending order, moving eigenvectors along with them
        
        #debugging...
        with open(os.path.join(outdir, 'eigenstuff.txt'), 'a') as outf:
            outf.write(str(estuff)) #write resulting intensity array to file
            outf.write('\n\n')
        
        
        A_er[i][j] = (1 - abs(estuff[0][0]/estuff[-1][0])) * intensities[i][j] #populate A_er matrix using appropriate formula and weighting
        #eigenvalues should already be working
        

with open(os.path.join(outdir, 'aniso_ratios.txt'), 'w') as outf:
    outf.write(str(A_er)) #write resulting intensity array to file

with open(os.path.join(outdir, 'mean_intensities.txt'), 'w') as outf:
    outf.write(str(intensities)) #write resulting intensity array to file
    

    

print("Done!")

#Perform anisotropy analysis at every pixel of the image
# im_d = Image.eval(im, anisotropy_analyze)

