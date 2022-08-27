'''
unit tests for downsample v2
'''
 
from hp.np import apply_blockwise_ufunc, apply_blockwise, dropna
from hp.rio import RioWrkr, write_array, load_array
from numpy import array, dtype
from tests2.conftest import validate_dict, src_dir, get_abs
import numpy as np
import pytest, copy, os, random
import rasterio as rio
xfail = pytest.mark.xfail


#===============================================================================
# helpers and globals------
#===============================================================================
crsid = 2953
prec=5

#for test data
output_kwargs = dict(crs=rio.crs.CRS.from_epsg(crsid),
                     transform=rio.transform.from_origin(1,100,1,1)) 

 
@pytest.fixture(scope='function')
def dem_fp(dem_ar, tmp_path): 
    return write_array(dem_ar, os.path.join(tmp_path, 'dem.tif'), **output_kwargs)
 

@pytest.fixture(scope='function')
def wse_fp(wse_ar, tmp_path): 
    return write_array(wse_ar, os.path.join(tmp_path, 'wse.tif'), **output_kwargs)