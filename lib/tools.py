# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:06:17 2016

@author: David
"""

import numpy

#
#class roi:
#    def __init__(self, ori, coh, ener, anir):
#        roi.orientation = ori
#        roi.coherence = coh
#        roi.energy = ener
#        roi.aniso_ratio = anir




def create_windowed_roi(input,startx,starty,width,height):
    """From a 2D array (input) containing brightness information of the original image, create a 2D array starting at the specified coordinates with the specified thicknesses, convoluted with a window function. Then, return the 2D array, normalized by having each element be the prior element's residual.
    The window function will be, by default, a Hamming function. (TODO: Look into other window functions.)"""

    roi = numpy.ndarray([height, width]) #2D arrays always have the left-to-right selector as the last coordinate
    
    #winy = numpy.hamming(height)
    #winx = numpy.hamming(width)
    
    # Rectangular Window
    winy = numpy.ones(height)
    winx = numpy.ones(width)

#    h = hamming(n)
#    ham2d = sqrt(outer(h,h))    
    
    
        
    mean = 0.0 #to be subtracted off each element

    roi = input[starty:starty+height, startx:startx+width]
    
    #convolute with window function
    for i in range(roi.shape[-1]):
        for j in range(roi.shape[-2]):
            roi[j][i] = roi[j][i]*winx[i]*winy[j]
            mean += roi[j][i]
#    
#    for j in range(width):
#        for i in range(height):
#            roi[j][i] = winx[j]*winy[i]*input[starty+j][startx+i] #create windowed value
#            mean += roi[j][i] #tally
#    
    mean /= roi.size
    
    for i in range(roi.shape[-1]):
        for j in range(roi.shape[-2]):
            roi[j][i] -= mean #replace original value with its residual, normalizing the array
    
    return roi
    

def calculate_relative_intensities(input, slice_numbers):
    """Input a n-dimentionsal array, and the number of subregions to be made (in a list of ints, n_slices; e.g., [2,3] would divide the original image into 6 rectangles, 2 fitting in the x-direction and 3 fitting in the y-direction). Return an n-dimensional array with the mean intensities of each subregion, relative to the mean intensity of the entire image.
    Will complain if the dimension of input does not match the size of slice_numbers."""
    
    # TODO: check that the resulting matrix isn't actually transposed from what it should be    
    
    if(input.ndim != len(slice_numbers)):
        print('The dimension of the input data array and the specified slice numbers do not match!')
        pass
    
    intensities = numpy.ndarray(slice_numbers) #creates container for subsequent calculations
    
    roi_sizes = []

    for i in range(len(input.shape)):
        roi_sizes.append(input.shape[i] / slice_numbers[i]) #get dimensions of ROI for each x,y,z,etc. component
    
    # roi_sizes = input.shape/slice_numbers #tuple/tuple division doesn't work :'(
    
    roi = numpy.ndarray(roi_sizes) #gives the roi its proper shape
    #TODO: check for rounding problems (since dividing ints by ints)
    
    total_sum = 0.0
    temp_sum = 0.0    
    
    #TODO: how to generalize looping properly?
    for i in range(slice_numbers[0]):
        for j in range(slice_numbers[1]): #for each roi
            roi = input[roi_sizes[0]*i:roi_sizes[0]*(i+1),roi_sizes[1]*j:roi_sizes[1]*(j+1)]        #pull out the information on a given roi. Note: comma between slices (not open/close brackets)
        
            for a in range(roi.shape[0]): #at each x-pixel within an roi
                for b in range(roi.shape[1]): #at each y-pixel within an roi
                    temp_sum += roi[a][b]
            
            total_sum += temp_sum
            
            intensities[i][j] = temp_sum / roi.size #mean intensity
            
            temp_sum = 0
    
    
    mean_intensity = total_sum / input.size #average across all pixels
    
    intensities /= mean_intensity #normalize intensities matrix by mean_intensity of entire image
    
    return intensities
    

def compute_power_matrix(input, n_bins):
    """ Takes in a matrix containing the complex values obtained from performing a Fourier Transform on a set of data, with redundant values (i.e. those values for frequencies above or equal to the Nyquist frequency) removed, input.
    Returns an 2-by-n_bins matrix, with each column denoting the x- and y- components of the power of all frequencies within a particular phase shift range (with thickness of 2*pi/(n_bins)).
    With respect to the phase calculated, the angle is shifted by -pi/2 to indicate the phase of the equivalent cosine function (as opposed to the 'default' sine representation). This is to show the angle at which the change in brightness is maximal (cosine) instead of minimal (sine)."""
    
    # For some reason, we stop caring about the spatial-frequency of the funcctions from which the phase shift and amplitude originated?    
        # This goes into the spirit of averaging over the entire region of interest. We're nont worried about local fluctuation within the ROI as a result of different spatial-frequency vavlues; we're more interested in understanding, when looking at the entire ROI, which direction corresponds to the most anisotropy, and how significantly different is that direction from its orthogonal direction?
      
    
    dtheta = 2*numpy.pi/n_bins
    power_sum = numpy.zeros([2,n_bins])
    
    
    #for every element in the frequency-representation of the original function

    it = numpy.nditer(input, flags = ['multi_index'])    
    
    for z in it:
        if z == input[0][0]:
            continue #ignore the "DC" value
        a = z.real
        b = z.imag
    
        #get the sinusoid's power
        power = a**2 + b**2
        

        if power == 0:
            #the sinusoid with the frequency specified by the (i,j) coordinates has no power
            # so it doesn't contribute to the power spectrum
            continue
        
        #get the sinusoid's phase
        phase = 0
        
        if a == 0:  #special case to avoid division by zero
            if b > 0:
                phase = numpy.pi * 1/2
            elif b < 0:
                phase = numpy.pi * 3/2
            #else:
            #    print('Fourier transform spat out a sinusoid term with no amplitude or phase!')
        else:
            phase = numpy.arctan(b/a)
        
        phase -= numpy.pi * 1/2 #shift from sine shift to cosine shift
        if phase < 0:
            phase += numpy.pi * 2 #shift to positive phase, to ensure proper placement into power-array
        
        #figure out which bin to place this element's power components; place it in
        i = phase / dtheta
        power_sum[0][i] += a**2 #cosine of the power; the x-component of the power
        power_sum[1][i] += b**2 #sine of the power; the y-component of the power
    
    return power_sum
    
    #now power_sum contains a measure of how much (the sine functions whose sum describe the original function) tended to be phase-shifted by a particular amount, weighted by their contribution to the image.
    # We can think of this as a measure of how much the sine functions wanted to be oriented in a particular x-y direction (as given by theta).
    # So we re-decompose the powers into their x- and y- components.
    # We can then obtain the covariance between the powers along the x- and y-components; this would describe how often a change in power along the x-direction gave rise to a change in power along the y-direction. This is essentially describing the anisotropy.
    # We can then use principal component analysis to obtain eigenvalues and eigenvectors describing the directions in which most of the variance in the power covariance matrix can be described.
    # We can look at the eigenvector corresponding to the dominant eigenvalue as the principal component, describing most of the covariance. This means it describes the most anisotropy, and its angle (obtained via the arctangent) describes the angle of most anisotropy
    

def coherence(v1,v2):
    """ Computes the coherence of two eigenvalues, v1 and v2. """
    
    return (max(v1,v2)-min(v1,v2))/(v1+v2)
        
        
def aniso_ratio(v1,v2,i):
    """ Computes the intensity-weighted anisotropic ratio from eigenvalues v1 and v2 and mean intensity i."""
    return (1 - min(v1,v2)/max(v1,v2)) * i

def perform_pca(M):
    """ Perform principal component analysis on the matrix M.
    Returns a list of tuples of (eigenvalue, eigenvector), sorted in ascending order of eigenvalue. (Eigenvectors have been normalized to unit vectors.)"""
    
    evals_arr, evecs_arr = numpy.linalg.eig(M)
    
    #TODO: clean up below code for normalizing eigenvectors?
    
    ezip = list(zip(evals_arr, evecs_arr))  #build up list of tuples of eignevalues and eigenvectors
    estuff = []

    for eval, evec in ezip:
        l = numpy.sqrt(sum(evec**2)) #length of eigenvectcor
        evec = evec / l #normalize eigenvectors to unit vectors       
        estuff.append((eval,evec))
    
    
    estuff = sorted(estuff, key= lambda tup: tup[0]) #sort eigenvalues in ascending order, moving eigenvectors along with them
    
    return estuff #still not entirely sure why I needed to "re-store" estuff before returning the sorted list, instead of just returning sorted(...) ...
    
    
def get_orientation(estuff):
    """ From a list of tuples of (eigenvalue, eigenvector), get the orientation of the eigenvector corresponding to the largest eigenvector. The orientation is returned in degrees, measured against the +x direction, and is bounded by [-pi, pi].
    PRECONDITION: the list of tuples (estuff) is already sorted in *ascending* order."""
    
    v = estuff[-1][1]
    x = v[0]
    y = v[1]
    
    theta = 0.0
    #deal with special cases
    if x == 0:
        if y > 0:
            theta = numpy.pi / 2
        elif y < 0:
            theta = -numpy.pi / 2
        else:
            err = 'An eigenvector was the zero vector!'
            print(err)
            return None
    else:
        theta = numpy.arctan(y/x) #v[0] and v[1] are x- and y- components
    
    # Because arctan is bounded as (-pi/2, pi/2), need to consider x- and y- components to recover the proper angle
    if (x < 0 and y > 0):
        theta += numpy.pi
    elif (x < 0 and y < 0):
        theta -= numpy.pi
    
    theta *= (180 / numpy.pi)
    
    if theta < 0 and theta > -90:
        theta += 90 #to account for the "pi/2 phase shift between Fourier space and Cartesian space"
        #TODO: *Why* is there a phase shift between Fourier space and Cartesian space?
    
    return theta