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

import numpy as np #for fast Fourier Transform
from matplotlib import pyplot
from PIL import Image
from lib import tools as t
#import json # to write roi struct to file


np.set_printoptions(precision=2, suppress = True) #for easier-to-look-at numbbers when printing in pdb

outdir = os.path.join(dname, 'outputs') #directory for output files

#Take in image file, as bytearray


im_orig,im_name = t.get_image(dep)
im = im_orig.convert('L') #convvert to grayscale

(xsize, ysize) = im.size #im is still an Image, and not an array


roix, roiy = t.get_ROI(im)


numx = int(xsize/roix)
numy = int(ysize/roiy)


# in grayscale: 0 = black, 255 = white
data = np.array(im)


#numx = 20
#numy = 20





#calculate unaltered mean intensities of the roi's, and the mean intensity of the entire image, to weight the anisotropy eigenvalue-ratio index later on

intensities = t.calculate_relative_intensities(input=data, slice_numbers=(numy, numx)) #have to list slice_numbers in "reverse" dimensional order

with open(os.path.join(outdir, 'eigenstuff.txt'), 'w') as outf:
    outf.write('Below are the eigenvalues and eigenvectors obtained for each region of interest. This analysis had ROIs that tiled the original image ' + str(numx) + ' times in the x-direction and ' + str(numy) + ' times in the y-direction.\n For more information, look into the comments of the Python script: \n\n\n')

A_er = np.ndarray(intensities.shape) #anisotropy ratio
C = np.ndarray(intensities.shape) #coherence
E = np.ndarray(intensities.shape) #energy
oris = np.ndarray(intensities.shape) #orientation. Saved as degrees
#to plot a vector field, it's easier to store the x- and y- components in separate arrays. Later on, will zip together
evecs_x = np.ndarray(intensities.shape)
evecs_y = np.ndarray(intensities.shape)

evecs = [[[] for x in range(intensities.shape[-1])] for y in range(intensities.shape[-2])] #2D array of arrays. Will have eigenvectors
roi_infos = [[{} for x in range(intensities.shape[-1])] for y in range(intensities.shape[-2])] #2D array for dicts. Will have labeled info

for i in range(0, numx):
    for j in range(0, numy):
        print('Working on x,y-slice (' + str(i+1) + ',' + str(j+1) + ')...')
        roi = t.create_windowed_roi(input=data, startx=i*roix, starty=j*roiy, width=roix, height=roiy) #take subsections, have them normalized
        roi_f = np.fft.fftn(roi, s=None, axes=None, norm=None) #perform discrete Fourier transform on ROI
        #could potentially do a real FFT/Hermitian FFT. Could save computational time?
        #TODO: Should we normalize? (i.e. make it 'ortho')
        
        # create radial histogram/profile ## TODO: Why? How?
        
        
        roi_f_g = roi_f[0:roi_f.shape[0]/2, 0:roi_f.shape[1]/2] #crop out the redundant negative ferquencies, and the Nyquist frequency (if the axis is even), to leave only the entries whose values have meaning
        
        n_b = 100 #number of bins to divide the frequencies up (since)
        
        #print('Before power matrix computation.')
        P = t.compute_power_matrix(input=roi_f_g, n_bins = n_b)
        #print('After power matrix computation.')
        
        cov = np.dot(P, np.transpose(P)) / P.shape[1] #obtain the covariance matrix, a 2-by-2 matrix for a 2D image
        #but how to normalize? Is this correct?
        #Also, I keep having to flip the order of multiplication... Something seems fishy...
        
        #evals,evecs = np.linalg.eig(cov)
        
        
        estuff = t.perform_pca(cov)        
        

        lambda_max = estuff[-1][0]
        lambda_min = estuff[0][0]
        
        coherence = t.coherence(lambda_max, lambda_min)
        energy = lambda_max + lambda_min
        
        #debugging...
        with open(os.path.join(outdir, 'eigenstuff.txt'), 'a') as outf:
            outf.write(str(estuff) + '\n') #write resulting intensity array to file
            #np.savetxt(fname = outf, X = estuff, delimiter = ' ')
            #outf.write('\n')
            outf.write('Coherence: ' + str(coherence) + '\n')
            outf.write('Energy: ' + str(energy) + '\n')
            outf.write('\n\n')
        
        #populate matrices
        A_er[j][i] = t.aniso_ratio(lambda_max, lambda_min, intensities[j][i]) #populate A_er matrix using appropriate formula and weighting
        C[j][i] = coherence
        E[j][i] = energy
        evec = t.rotate_vector(v = estuff[-1][1], theta = 90) #rotate vector by 90 degrees, to "compensate for pi/2 shift between Fourier space and Cartesian space"
        oris[j][i] = t.get_orientation(evec)
        #oris[j][i] = t.get_evec_orientation(estuff)
        evecs[j][i] = estuff[-1][1] #dominant eigenvector
        # break down into components; for plotting vector field
        evecs_x[j][i], evecs_y[j][i] = evec[0], evec[1]        
        #evecs_x[j][i], evecs_y[j][i] = estuff[-1][1][0], estuff[-1][1][1]

        # Lump together relevant info into a dictionary for each ROI
        roi_info = {'aniso_ratio': A_er[j][i], 'coherence': coherence, 'energy': energy, 'orientation': oris[j][i]}
        roi_infos[j][i] = roi_info

#evecs = list(zip(evecs_x, evecs_y))





with open(os.path.join(outdir, 'aniso_ratios.txt'), 'wb') as outf: #needs to be in binary form to use np.savetxt
    np.savetxt(fname = outf, X = A_er, fmt = '%10.5f', delimiter = ' ')
    #outf.write(str(A_er)) #write resulting intensity array to file

with open(os.path.join(outdir, 'mean_intensities.txt'), 'wb') as outf: #needs to be in binary form to use np.savetxt
    np.savetxt(fname = outf, X = intensities, fmt = '%10.5f', delimiter = ' ')
    #outf.write(str(intensities)) #write resulting intensity array to file

with open(os.path.join(outdir, 'orientatons.txt'), 'wb') as outf: #needs to be in binary form to use np.savetxt
    np.savetxt(fname = outf, X = oris, fmt = '%10.5f', delimiter = ' ')

with open(os.path.join(outdir, 'roi_infos.txt'), 'w') as outf: #needs to be in binary form to use np.savetxt
    outf.write(str(roi_infos))


evec_field = t.plot_vector_field(vecs_x = evecs_x, vecs_y = evecs_y, lens = [numx, numy], deltas = [roix, roiy])

print('A vector field of the eigenvectors derived from this analysis has been plotted, and orientation such that the vectors appear in the same relative location as the ROI it describes.')
plot_mark = input("Would you like to view and save this vector field? (Y/N):")

if plot_mark.lower() == 'y':
    pyplot.show(evec_field)

    pyplot.savefig(os.path.join(outdir,'evec_field.pdf'), bbox_inches='tight')
    print('The eigenvector field has been saved in the outputs directory in rasterized form.')
    
    evecfield_fp = os.path.join(outdir, 'evec_field.png')
    pyplot.savefig(evecfield_fp, bbox_inches='tight')
    
    with Image.open(evecfield_fp) as evec_field_img:
        merged = t.overlay_images(foreground = evec_field_img, background = im_orig) #currently broken...
        merged.save(os.path.join(outdir,'merged.png'), 'PNG')


## TODO: allow user to select what information to use for HSB/HSV
# hues == orientations, saturations == coherencies, values == A_er

hsv = t.create_hsv_array(hues = oris, saturations = A_er, values = None, original_image = im_orig)
rgb = t.hsv_to_rgb(hsv)

#scale the [0,1] float rgb values to [0,255] ints

rgb = np.array(255*rgb, dtype = int)

#fci = Image.fromarray(hsv, mode = 'HSV') #false-colored image
fci = Image.fromarray(rgb, mode = 'RGB') #false-colored image
fci = fci.resize(im_orig.size) #scale back up to original_image size

fci.save(os.path.join(outdir, im_name+'_analyzed (xsize=' + str(numx) + ',ysize=' + str(numy) + ').jpg'))

print("A falsely colored image has been created.") ## TODO: Make this more meaningful...

print("Done!")



