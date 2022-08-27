'''
Created on Aug. 27, 2022

@author: cefect
'''
import numpy as np
def get_rand_ar(shape, scale=10):
    dem_ar =  np.random.random(shape)*scale
    wse_ar = get_wse_filtered(np.random.random(shape)*scale, dem_ar)
    
    return dem_ar, wse_ar

def get_wse_filtered(wse_raw_ar, dem_ar):
    """mask out negative WSE values"""
    wse_ar = wse_raw_ar.copy()
    np.place(wse_ar, wse_raw_ar<=dem_ar, np.nan)
    
    return wse_ar