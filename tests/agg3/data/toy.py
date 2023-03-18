'''
Created on Dec. 5, 2022

@author: cefect

toy test data
'''
import numpy as np
import pandas as pd
import numpy.ma as ma
import shapely.geometry as sgeo
from pyproj.crs import CRS
#from tests.conftest import get_rlay_fp
nan, array = np.nan, np.array

"""setup to construct when called
seems cleaner then building each time we init (some tests dont need these)
more seamless with using real data in tests"""

#===============================================================================
# helpers
#===============================================================================
from hp.tests.tools.rasters import (
    get_mar, get_ar_from_str, get_wse_ar, get_rlay_fp,crs_default, 
    )
from hp.hyd import get_wsh_ar



#===============================================================================
# raw data
#===============================================================================


dem1_ar = get_mar(
    get_ar_from_str("""
    1    1    1    9    9    9    
    1    1    1    9    9    9
    1    1    1    2    2    9
    2    2    2    9    2    9
    6    2    2    9    2    9
    2    2    2    9    2    9
    4    4    4    2    2    9
    4    4    4    9    9    9
    4    4    4    9    1    1
    4    4    4    2    2    9
    4    4    4    9    9    9
    4    4    4    9    1    1
    """))


 

"""dummy validation against wse1_ar3
1FP, 1FN"""
wse1_ar = get_mar(
    array([
        [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3.,  3.,  3., nan],
       
       [ 4.,  4.,  4., nan,  3., nan],
       [nan,  4.,  4., nan,  3., nan],
       [ 4.,  4.,  4., nan,  3., nan],
       
       [ 5.,  5.,  5.,  5.,  5., nan],
       [ 5.,  5.,  5., nan, nan, 5.],
       [ 5.,  5.,  5., nan, nan, nan],
       
       [ 5.,  5.,  5., nan,  nan, nan],
       [ 5.,  5.,  5., nan, nan, nan],
       [ 5.,  5.,  5., nan, nan, nan]
       
       ])
    )

 
    
wsh1_ar = get_wsh_ar(dem1_ar, wse1_ar)

 
#==============================================================================
# agg data
#==============================================================================
#standin for hydrodynamic coarse results
wse2_ar = get_mar( #get_wse_ar converts 9999 to null. get_mar converts back to -9999
    get_wse_ar("""
    3.5    -9999
    4    -9999
    5    -9999
    5    -9999
    """))

s = dem1_ar.shape
bbox_default = sgeo.box(0, 0, s[1], s[0]) #(0.0, 0.0, 6.0, 12.0)
#print(f'toy bbox_default={bbox_default.bounds}')

