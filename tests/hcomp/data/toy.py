'''
Created on Dec. 5, 2022

@author: cefect

toy test data
'''
import numpy as np
import pandas as pd
import numpy.ma as ma
from io import StringIO
#from tests.conftest import get_rlay_fp
nan, array = np.nan, np.array

"""setup to construct when called
seems cleaner then building each time we init (some tests dont need these)
more seamless with using real data in tests"""

#===============================================================================
# helpers
#===============================================================================
from hp.tests.tools.rasters import get_mar, get_ar_from_str, get_wse_ar, get_rlay_fp
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

 




