'''
Created on Aug. 27, 2022

@author: cefect

commons for Haz
'''
import copy
import numpy as np
import numpy.ma as ma
import pandas as pd

#===============================================================================
# globals
#===============================================================================
#integer maps for buildilng the mosaic
cm_int_d = {'DD':11, 'WW':21, 'WP':31, 'DP':41}

#stat dx labels (and values) order matters!
index_names = ['scale', 'pixelArea', 'pixelLength']
 
coldx_d = { 
    'layer':['dem', 'wse', 'wd', 'catMask', 'diff'],

    'dsc':list(cm_int_d.keys())+['all'],
    'metric':['mean', 'posi_area', 'post_count', 'pre_count', 'vol'],
    }
    
 


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
        
 

        
        if not np.any(ar.mask):
            raise AssertionError('expect some masked values on the wse\n'+msg)
        
        if np.all(ar.mask):
            raise AssertionError('expect some unmasked values on the wse\n'+msg)
        
        if not np.all(ar>0):
            raise AssertionError('got some negatives\n'+msg)
        
 
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
        
def assert_dx_names(dx, msg=''):
    """index name expectations"""
    if not __debug__: # true if Python was not started with an -O option
        return
 
    __tracebackhide__ = True
    
    if not isinstance(dx, pd.DataFrame):
        raise TypeError('bad type: %s\n%s'%(type(dx).__name__, msg))
 
    #===========================================================================
    # check names
    #===========================================================================
    for axis, vali_l, test_l in (
        ('index', copy.deepcopy(index_names),  list(dx.index.names)),
        ('columns', list(coldx_d.keys()),  list(dx.columns.names)),
        ):
        if not np.array_equal(np.array(vali_l), np.array(test_l)):
            raise AssertionError('%s does not match name expectations: \n    %s\n    %s'%(axis, test_l, msg))
        
    #===========================================================================
    # check values
    #===========================================================================
    mdex = dx.columns
    
    for name in mdex.names:
        miss_l = set(mdex.get_level_values(name)).difference(coldx_d[name])
        assert len(miss_l)==0, 'on %s got %i unrecognized values: %s\n'%(
            name, len(miss_l), miss_l)+msg
 
    
    #dx.columns.get_level_values(''
    
    
    
    
    
    
    
    
    
    
        