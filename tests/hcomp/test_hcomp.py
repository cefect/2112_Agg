'''
Created on Mar. 18, 2023

@author: cefect

tests for the hydro-downscale compare
'''

import pytest, copy, os, random, re
xfail = pytest.mark.xfail
import numpy as np
import pandas as pd
import shapely.geometry as sgeo
from rasterio.enums import Resampling

from hcomp.scripts import HydCompareSession
from tests.hcomp.conftest import get_aoi_fp
from hp.tests.tools.rasters import get_rlay_fp
#===============================================================================
# test data------
#===============================================================================
 
from tests.hcomp.data.toy import dem1_ar, wse1_ar, wsh1_ar

dem1_fp = get_rlay_fp(dem1_ar, 'dem1') 
 
wse1_fp = get_rlay_fp(wse1_ar, 'wse12')
wsh1_fp = get_rlay_fp(wsh1_ar, 'wsh1')
 
aoi_fp = get_aoi_fp(sgeo.box(0, 30, 60, 60))



#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name, 
                    ):
    
    """Mock session for tests"""
 
    #np.random.seed(100)
    #random.seed(100)
    
    with HydCompareSession(  
 
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                 #prec=prec,
                  proj_name='test', #probably a better way to propagate through this key 
                 run_name=test_name[:8].replace('_',''),
                  
                 relative=True, write=write, #avoid writing prep layers
                 
                 logger=logger, overwrite=True, 

                 
 
                   ) as ses:
 
        yield ses

#===============================================================================
# tests-------
#===============================================================================
@pytest.mark.parametrize('dem_fp, wsh_fp', [
    (dem1_fp,wsh1_fp),
    ])
@pytest.mark.parametrize('aggscale', [3])
@pytest.mark.parametrize('resampling', [Resampling.average])
def test_get_avgWSH(wsh_fp, dem_fp, aggscale, resampling, wrkr): 
    wrkr.get_avgWSH(wsh_fp, dem_fp, aggscale=aggscale, resampling=resampling)
    
    
#===============================================================================
# @pytest.mark.parametrize('dem_fp, wse_fp', [
#     (dem1_fp,wse1_fp),
#     ])
#===============================================================================
