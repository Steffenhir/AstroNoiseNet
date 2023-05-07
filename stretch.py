# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:44:29 2022

@author: steff
"""



import numpy as np




def stretch_channel(o, s, c, bg, sigma, median, mad):

    o_channel = o[:,:,c]
    s_channel = s[:,:,c]


    shadow_clipping = np.clip(median - sigma*mad, 0, 1.0)
    highlight_clipping = 1.0

    midtone = MTF((median-shadow_clipping)/(highlight_clipping - shadow_clipping), bg)

    o_channel[o_channel <= shadow_clipping] = 0.0
    o_channel[o_channel >= highlight_clipping] = 1.0
    
    s_channel[s_channel <= shadow_clipping] = 0.0
    s_channel[s_channel >= highlight_clipping] = 1.0

    indx_inside_o = np.logical_and(o_channel > shadow_clipping, o_channel < highlight_clipping)
    indx_inside_s = np.logical_and(s_channel > shadow_clipping, s_channel < highlight_clipping)

    o_channel[indx_inside_o] = (o_channel[indx_inside_o]-shadow_clipping)/(highlight_clipping - shadow_clipping)
    s_channel[indx_inside_s] = (s_channel[indx_inside_s]-shadow_clipping)/(highlight_clipping - shadow_clipping)

    o_channel = MTF(o_channel, midtone)
    s_channel = MTF(s_channel, midtone)
    o[:,:,c] = o_channel[:,:]
    s[:,:,c] = s_channel[:,:]


def stretch(o, s, bg, sigma, median, mad):

    o_copy = np.copy(o)
    s_copy = np.copy(s)
    
    for c in range(o_copy.shape[-1]):
        stretch_channel(o_copy, s_copy, c, bg, sigma, median[c], mad[c])

    return o_copy, s_copy


def stretch_channel_single(o, c, bg, sigma, median, mad):

    o_channel = o[:,:,c]


    shadow_clipping = np.clip(median - sigma*mad, 0, 1.0)
    highlight_clipping = 1.0

    midtone = MTF((median-shadow_clipping)/(highlight_clipping - shadow_clipping), bg)

    o_channel[o_channel <= shadow_clipping] = 0.0
    o_channel[o_channel >= highlight_clipping] = 1.0
    

    indx_inside = np.logical_and(o_channel > shadow_clipping, o_channel < highlight_clipping)

    o_channel[indx_inside] = (o_channel[indx_inside]-shadow_clipping)/(highlight_clipping - shadow_clipping)

    o_channel = MTF(o_channel, midtone)
    o[:,:,c] = o_channel[:,:]


def stretch_channel_single_inverse(o, c, bg, sigma, median, mad):
    o_channel = o[:,:,c]
    
    shadow_clipping = np.clip(median - sigma*mad, 0, 1.0)
    highlight_clipping = 1.0
    midtone = MTF((median-shadow_clipping)/(highlight_clipping - shadow_clipping), bg)
    
    o_channel = MTF_inverse(o_channel, midtone)
    
    o_channel = o_channel * (highlight_clipping - shadow_clipping) + shadow_clipping
    
    o[:,:,c] = o_channel[:,:]
    


def stretch_single(o, bg, sigma, median, mad):

    o_copy = np.copy(o)
    
    for c in range(o_copy.shape[-1]):
        stretch_channel_single(o_copy, c, bg, sigma, median[c], mad[c])

    return o_copy

def stretch_single_inverse(o, bg, sigma, median, mad):

    o_copy = np.copy(o)
    
    for c in range(o_copy.shape[-1]):
        stretch_channel_single_inverse(o_copy, c, bg, sigma, median[c], mad[c])

    return o_copy

def MTF(data, midtone):

    if type(data) is np.ndarray:
        data[:] = (midtone-1)*data[:]/((2*midtone-1)*data[:]-midtone)
    else:
        data = (midtone-1) * data / ((2*midtone-1) * data - midtone)

    return data

def MTF_inverse(data, midtone):
    if type(data) is np.ndarray:
        data[:] = midtone*data[:]/((2*midtone-1)*data[:]-(midtone-1))
    else:
        data = midtone * data / ((2*midtone-1) * data - (midtone-1))

    return data

    
