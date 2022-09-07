'''
Created on Sep. 6, 2022

@author: cefect
'''

from tests2.conftest import validate_dict, src_dir, get_abs
import numpy as np
import pandas as pd
import pytest, copy, os, random
from agg2.expo.scripts import ExpoSession as Session
import shapely.geometry as sgeo
from pyproj.crs import CRS
#===============================================================================
# FIXTURES-----
#===============================================================================
bbox1 = sgeo.box(7, 88, 15, 97)


@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name, 
                    ):
    
    """Mock session for tests"""
 
    np.random.seed(100)
    random.seed(100)
 
    
    with Session(  
                 #GeoPandas
                 crs=CRS.from_user_input(2953),
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                 #prec=prec,
                  proj_name='expoTest', #probably a better way to propagate through this key 
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

@pytest.mark.parametrize('pick_fp', [r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data\cMasks\dsTest_test02_0829_haz_cMasks.pkl'])
@pytest.mark.parametrize('finv_fp', [r'C:\LS\09_REPOS\02_JOBS\2112_agg\cef\tests2\expo\data\finv_SJ_test_0906.geojson'])

@pytest.mark.parametrize('bbox', [
                                bbox1, 
                                #None
                                  ])
def test_01_assetRsc(wrkr, pick_fp, finv_fp, bbox):
    raise IOError('need to make this fit better within the raster bounds')
    rdx = wrkr.build_assetRsc(pick_fp, finv_fp, bbox=bbox)
    
    #validate
    assert isinstance(rdx, pd.DataFrame)
    assert isinstance(rdx.columns, pd.MultiIndex)
    assert len(rdx.columns.names)==2
    
    
    
    
    
    
    
    