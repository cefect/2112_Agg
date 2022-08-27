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
from agg2.haz.misc import get_rand_ar, get_wse_filtered
from agg2.haz.scripts import Dsamp
xfail = pytest.mark.xfail


#===============================================================================
# helpers and globals------
#===============================================================================
crsid = 2953
prec=5

#for test data
output_kwargs = dict(crs=rio.crs.CRS.from_epsg(crsid),
                     transform=rio.transform.from_origin(1,100,1,1)) 

 


#===============================================================================
# FIXTURES-----
#===============================================================================
@pytest.fixture(scope='function')
def dem_fp(dem_ar, tmp_path): 
    return write_array(dem_ar, os.path.join(tmp_path, 'dem.tif'), **output_kwargs)
 

@pytest.fixture(scope='function')
def wse_fp(wse_ar, tmp_path): 
    return write_array(wse_ar, os.path.join(tmp_path, 'wse.tif'), **output_kwargs)

@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name, 
                    ):
    
    """Mock session for tests"""
 
    np.random.seed(100)
    random.seed(100)
 
    
    with Dsamp(  
                 
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                 prec=prec,
                  proj_name='dsTest', #probably a better way to propagate through this key 
                 run_name=test_name[:8].replace('_',''),
                  
                 relative=True, write=write, #avoid writing prep layers
                 
                 logger=logger, overwrite=True,
                   
                   #oop.Session
                   exit_summary=False,logfile_duplicate=False,
                   compiled_fp_d=dict(), #I guess my tests are writing to the class... no tthe instance
 
                   ) as ses:
        
 
        assert len(ses.data_d)==0
        assert len(ses.compiled_fp_d)==0
        assert len(ses.ofp_d)==0
        yield ses

#===============================================================================
# UNIT TESTS--------
#===============================================================================


@pytest.mark.parametrize('base_resolution', [1, 3])
@pytest.mark.parametrize('reso_iters', [3, 10])
def test_00_downSampleIter(wrkr, base_resolution, reso_iters):
    
    #build with function
    res_l = wrkr.get_downSampleIter(base_resolution=base_resolution, reso_iters=reso_iters)
    
    #===========================================================================
    # #validate
    #===========================================================================
    assert isinstance(res_l, list)
    assert len(res_l)==reso_iters
    assert res_l[0]==base_resolution
 
@pytest.mark.dev 
@pytest.mark.parametrize('dem_ar, wse_ar', [
    get_rand_ar((4,6))
    ])
@pytest.mark.parametrize('method', ['direct', 'filter'])
@pytest.mark.parametrize('dsc_l', [([1,2,4])])
def test_01_dset(dem_fp,dem_ar,wse_fp, wse_ar,   wrkr, dsc_l, method):
    wrkr.build_dset(dem_fp, wse_fp, dsc_l=dsc_l, method=method)
