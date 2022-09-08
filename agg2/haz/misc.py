'''
Created on Aug. 27, 2022

@author: cefect
'''
import numpy as np
import numpy.ma as ma


def get_rand_ar(shape, scale=10):
    dem_ar =  np.random.random(shape)*scale
    wse_ar = get_wse_filtered(np.random.random(shape)*scale, dem_ar)
    
    return dem_ar, wse_ar

def get_wse_filtered(wse_raw_ar, dem_ar):
    """mask out negative WSE values"""
    wse_ar = wse_raw_ar.copy()
    np.place(wse_ar, wse_raw_ar<=dem_ar, np.nan)
    
    return wse_ar

def assert_wse_ar(ar, masked=False, msg=''):
    """check wse array satisfies assumptions"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    __tracebackhide__ = True
    
  
    assert 'float' in ar.dtype.name
    
    if not masked:
        if isinstance(ar, ma.MaskedArray):
            raise AssertionError('masked=False but got a masked array')
        
        if not np.all(np.nan_to_num(ar, nan=9999)>0):
            raise AssertionError('got some negatives\n'+msg)
        
        if not np.any(np.isnan(ar)):
            raise AssertionError('expect some nulls on the wse\n'+msg)
        
    else:
        if not isinstance(ar, ma.MaskedArray):
            raise AssertionError('masked=True but got an unmasked array')
        
        if not np.all(ar>0):
            raise AssertionError('got some negatives\n'+msg)
        
        if not np.any(ar.mask):
            raise AssertionError('expect some masked values on the wse\n'+msg)
        
        if np.all(ar.mask):
            raise AssertionError('expect some unmasked values on the wse\n'+msg)
        
 
def assert_dem_ar(ar, masked=False, msg=''):
    """check DEM array satisfies assumptions"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    __tracebackhide__ = True
    
    assert isinstance(ar, np.ndarray) 
    assert 'float' in ar.dtype.name
    
    """relaxing this
    if not np.all(ar>0):
        raise AssertionError('got some negatives\n'+msg)"""
    
    if not masked:
        if isinstance(ar, ma.MaskedArray):
            raise AssertionError('masked=False but got a masked array')
        
        if not np.all(~np.isnan(ar)):
            raise AssertionError('got some nulls on DEM\n'+msg)
        
    else:
        if not isinstance(ar, ma.MaskedArray):
            raise AssertionError('masked=True but got an unmasked array')
        
        if np.any(ar.mask):
            raise AssertionError('expect no masks on the DEM\n'+msg)
        