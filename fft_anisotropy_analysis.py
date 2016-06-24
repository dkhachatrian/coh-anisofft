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
#import json # to write roi struct to file


numpy.set_printoptions(precision=2, suppress = True) #for easier-to-look-at numbbers when printing in pdb

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

with open(os.path.join(outdir, 'eigenstuff.txt'), 'w') as outf:
    outf.write('Below are the eigenvalues and eigenvectors obtained for each region of interest. This analysis had ROIs that tiled the original image ' + str(numx) + ' times in the x-direction and ' + str(numy) + ' times in the y-direction.\n For more information, look into the comments of the Python script: \n\n\n')

A_er = numpy.ndarray(intensities.shape) #anisotropy ratio
C = numpy.ndarray(intensities.shape) #coherence
E = numpy.ndarray(intensities.shape) #energy
oris = numpy.ndarray(intensities.shape) #orientation
evecs = [[[] for x in range(intensities.shape[-1])] for y in range(intensities.shape[-2])] #2D array of arrays. Will have eigenvectors
roi_infos = [[{} for x in range(intensities.shape[-1])] for y in range(intensities.shape[-2])] #2D array for dicts. Will have labeled info

for i in range(0, numx):
    for j in range(0, numy):
        print('Working on x,y-slice (' + str(i) + ',' + str(j) + ')...')
        roi = t.create_windowed_roi(input=data, startx=i*roix, starty=j*roiy, width=roix, height=roiy) #take subsections, have them normalized
        roi_f = numpy.fft.fftn(roi, s=None, axes=None, norm=None) #perform discrete Fourier transform on ROI
        #could potentially do a real FFT/Hermitian FFT. Could save computational time?
        #TODO: Should we normalize? (i.e. make it 'ortho')
        
        # create radial histogram/profile ## TODO: Why? How?
        
        
        roi_f_g = roi_f[0:roi_f.shape[0]/2, 0:roi_f.shape[1]/2] #crop out the redundant negative ferquencies, and the Nyquist frequency (if the axis is even), to leave only the entries whose values have meaning
        
        n_b = 100 #number of bins to divide the frequencies up (since)
        
        #print('Before power matrix computation.')
        P = t.compute_power_matrix(input=roi_f_g, n_bins = n_b)
        #print('After power matrix computation.')
        
        cov = numpy.dot(P, numpy.transpose(P)) / P.shape[1] #obtain the covariance matrix, a 2-by-2 matrix for a 2D image
        #but how to normalize? Is this correct?
        #Also, I keep having to flip the order of multiplication... Something seems fishy...
        
        #evals,evecs = numpy.linalg.eig(cov)
        
        
        estuff = t.perform_pca(cov)        
        

        lambda_max = estuff[-1][0]
        lambda_min = estuff[0][0]
        
        coherence = t.coherence(lambda_max, lambda_min)
        energy = lambda_max + lambda_min
        
        #debugging...
        with open(os.path.join(outdir, 'eigenstuff.txt'), 'a') as outf:
            outf.write(str(estuff) + '\n') #write resulting intensity array to file
            #numpy.savetxt(fname = outf, X = estuff, delimiter = ' ')
            #outf.write('\n')
            outf.write('Coherence: ' + str(coherence) + '\n')
            outf.write('Energy: ' + str(energy) + '\n')
            outf.write('\n\n')
        
        #populate matrices
        A_er[i][j] = t.aniso_ratio(lambda_max, lambda_min, intensities[i][j]) #populate A_er matrix using appropriate formula and weighting
        C[i][j] = coherence
        E[i][j] = energy
        oris[i][j] = t.get_evec_orientation(estuff)
        evecs[i][j] = estuff[-1][1] #dominant eigenvector

        # Lump together relevant info into a dictionary for each ROI
        roi_info = {'aniso_ratio': A_er[i][j], 'coherence': coherence, 'energy': energy, 'orientation': oris[i][j]}
        roi_infos[i][j] = roi_info


t.plot_vector_field(evec_arr = evecs, deltas = [roix, roiy])



with open(os.path.join(outdir, 'aniso_ratios.txt'), 'wb') as outf: #needs to be in binary form to use numpy.savetxt
    numpy.savetxt(fname = outf, X = A_er, fmt = '%10.5f', delimiter = ' ')
    #outf.write(str(A_er)) #write resulting intensity array to file

with open(os.path.join(outdir, 'mean_intensities.txt'), 'wb') as outf: #needs to be in binary form to use numpy.savetxt
    numpy.savetxt(fname = outf, X = intensities, fmt = '%10.5f', delimiter = ' ')
    #outf.write(str(intensities)) #write resulting intensity array to file

with open(os.path.join(outdir, 'orientatons.txt'), 'wb') as outf: #needs to be in binary form to use numpy.savetxt
    numpy.savetxt(fname = outf, X = oris, fmt = '%10.5f', delimiter = ' ')

with open(os.path.join(outdir, 'roi_infos.txt'), 'w') as outf: #needs to be in binary form to use numpy.savetxt
    outf.write(str(roi_infos))


print("Done!")

#Perform anisotropy analysis at every pixel of the image
# im_d = Image.eval(im, anisotropy_analyze)

