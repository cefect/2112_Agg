'''
Created on Feb. 22, 2022

@author: cefect

test for stochastic model
'''
import os  
import pytest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
idx = pd.IndexSlice

import numpy as np
np.random.seed(100)
from numpy.testing import assert_equal

from agg.hyd.scripts import ModelStoch
from tests.conftest import retrieve_finv_d, retrieve_data, search_fp

@pytest.fixture
def modelstoch(tmp_path,
            #wrk_base_dir=None, 
            base_dir, write,logger, feedback,#see conftest.py (scope=session)
            iters=5,
            proj_lib =     {
                    #===========================================================
                    # 'point':{
                    #       'EPSG': 2955, 
                    #      'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0218.geojson', 
                    #      'dem': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\dem_obwb_test_0218.tif', 
                    #      'wd_dir': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd',
                    #      #'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                    #         }, 
                    #===========================================================
                    'testSet1':{
                          'EPSG': 2955, 
                         'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0219_poly.geojson', 
                         'dem': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\dem_obwb_test_0218.tif', 
                         'wd_fp_d': {'hi':r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd\wd_test_0218.tif'},
                         #'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                            }, 
                        },
                    ):
    """
    TODO: get module scoped logger
    """
    
    #get working directory
    wrk_dir = None
    if write:
        wrk_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    
    with ModelStoch(out_dir = tmp_path, proj_lib=proj_lib, wrk_dir=wrk_dir, 
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     iters=iters,
                     ) as ses:
        yield ses
        
@pytest.mark.dev
@pytest.mark.parametrize('tval_type',['rand', 'uniform'], indirect=False) #uniform is somewhat silly here 
@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
def testS_tvals(modelstoch,tval_type, finv_agg_fn, true_dir, base_dir, write):
    #===========================================================================
    # load inputs   
    #===========================================================================
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, modelstoch, base_dir)
    
    #===========================================================================
    # execute
    #===========================================================================
    dkey='tvals'
    tv_dx = modelstoch.build_tvals(dkey=dkey, tval_type=tval_type, 
                            finv_agg_d=finv_agg_d, mindex =finv_agg_mindex, write=write)
    
    #===========================================================================
    # retrieve true
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, modelstoch)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_frame_equal(tv_dx, true)
    
    
    
    
    
    
    