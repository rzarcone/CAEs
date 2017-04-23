from __future__ import division
import numpy as np

from scipy import stats, interpolate

class Distribution(object):
    """
    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete inverstion of a cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distibution object returns indices into density array
    """
    def __init__(self, pdf, sort = True, interpolation = True, transform = lambda x: x):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform

        #a pdf can not be negative
#        assert(np.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        #construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)
    @property
    def ndim(self):
        return len(self.shape)
    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is imlpicitly normalized"""
        return self.cdf[-1]
    def __call__(self, N):
        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high = self.sum, size = N)
        #find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        #map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)
        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = index + np.random.uniform(size=index.shape)
        return self.transform(index)



def sample_arb_dist(Pyx, x, y, x_in, N=1):
    """
    Draw samples
    
    Pyx
        [n_outputs, n_inputs]
        Sorted low to high in BOTH dimensions

    x_in
        ndarray (n_samples,)

    """
    N=int(N)

    x = np.sort(x)
    y = np.sort(y)
    
    x_in = np.r_[x_in]
    x_in_shape = x_in.shape
    x_in = x_in.ravel()

    samples = np.zeros((x_in.size, N))
    idx = find_nearest(x, x_in)
    
    for i_, idx_, in enumerate(idx):
        dist = Distribution(Pyx[:,idx_], interpolation=False)
        index = dist(N).squeeze()
        samples[i_, :] = y[index]
    
    return samples.reshape(*x_in_shape)


def find_nearest(array, values, axis=0):
    if array.squeeze().ndim > 1:
       
       if axis == 0:
            idx = np.abs( array - np.c_[values.squeeze()] ).argmin(1)
       else:
            idx = np.abs( array - np.r_[values.squeeze()] ).argmin(0)
   
    else:
        idx = np.abs( np.subtract.outer(array, values) ).argmin(0)

    return idx
    

def device_noise(inputs, Pyx, in_range=[], out_range=[], x=None, y=None):
    '''Add device noise in a normalized manner
    '''
    if not in_range:
        in_range = (np.min(inputs), np.max(inputs))
    if not out_range:
        out_range = in_range
    if not x:
        x = np.linspace(in_range[0], in_range[1], Pyx.shape[1])
    if not y:
        y = np.linspace(out_range[0], out_range[1], Pyx.shape[0])
        
    # input_x = scale(inputs, in_range, out_range )
    # print input_x
    outputs = sample_arb_dist(Pyx, x, y, inputs)
    # print output_y
    # outputs = scale(output_y, out_range, in_range)

    return outputs
        
    


def scale(val, in_range, out_range, clip=True):
    '''Scale between in_range to out_range
    '''
    in_min, in_max = in_range
    out_min, out_max = out_range

    if type(val) is not np.ndarray:
    	val = np.r_[val]

    if clip:
        if in_min < in_max:
            val[val > in_max] = in_max
            val[val < in_min] = in_min
        else:
            val[val < in_max] = in_max
            val[val > in_min] = in_min
            

    output = out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

    return output
    

 