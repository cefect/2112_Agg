'''
Created on Aug. 27, 2022

@author: cefect
'''
import os, shutil, pickle
import pytest
import numpy as np
import pandas as pd
from definitions import src_dir
import shapely.geometry as sgeo
from pyproj.crs import CRS
from hp.logr import get_new_file_logger, get_new_console_logger, logging
from hp.rio import write_array

#===============================================================================
# VARS--------
#===============================================================================
bbox_base = sgeo.box(0, 0, 100, 100)
crs=CRS.from_user_input(2953)

#saint Jon sub-set test data
SJ_test_dir= r'C:\LS\10_OUT\2112_Agg\ins\hyd\SaintJohn\test'

proj_d = {
    'EPSG':2953,
    'finv_fp':r'C:\LS\09_REPOS\02_JOBS\2112_agg\cef\tests2\expo\data\finv_SJ_test_0906.geojson',
    'wse_fp_d':{'hi':os.path.join(SJ_test_dir,  'GeoNB_LSJ_aoiT01_0829.tif')},
    'dem_fp_d':{1:os.path.join(SJ_test_dir,'NBDNR2015_r01_aoiT01_0829.tif')},
    }
#===============================================================================
# MISC----
#===============================================================================
@pytest.fixture(scope='session')
def write():
    write=False
    if write:
        print('WARNING!!! runnig in write mode')
    return write

@pytest.fixture(scope='function')
def test_name(request):
    return request.node.name.replace('[','_').replace(']', '_')

@pytest.fixture(scope='session')
def logger():
    return get_new_console_logger(level=logging.DEBUG)


#===============================================================================
# DIRECTOREIES--------
#===============================================================================

@pytest.fixture
def true_dir(write, tmp_path, base_dir):
    true_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    if write:
        if os.path.exists(true_dir): 
            shutil.rmtree(true_dir)
            os.makedirs(true_dir) #add back an empty folder
            
    return true_dir

@pytest.fixture(scope='function')
def out_dir(write, tmp_path, tgen_dir):
    if write:
        return tgen_dir
    else:
        return tmp_path
    
 

def get_abs(relative_fp):
    return os.path.join(src_dir, relative_fp)

#===============================================================================
# test data------
#===============================================================================


#===============================================================================
# VALIDATIOn-------
#===============================================================================

def compare_dicts(dtest, dtrue, index_l = None, msg=''
                 ):
        df1 = pd.DataFrame.from_dict({'true':dtrue, 'test':dtest})
        
        if index_l is None:
            index_l = df1.index.tolist()
        
        df2 = df1.loc[index_l,: ].round(3)
        
        bx = ~df2['test'].eq(other=df2['true'], axis=0)
        if bx.any():
            raise AssertionError('%i/%i raster stats failed to match\n%s\n'%(bx.sum(), len(bx), df2.loc[bx,:])+msg)
                
def validate_dict(session, valid_dir, test_stats_d, baseName='base'):
    
    true_fp = os.path.join(valid_dir, '%s_true.pkl' % (baseName))
    
    #===========================================================================
    # write trues
    #===========================================================================
    
    if session.write:
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)
        with open(true_fp, 'wb') as f:
            pickle.dump(test_stats_d, f, pickle.HIGHEST_PROTOCOL)
    else:
        assert os.path.exists(true_fp)
        
    #===========================================================================
    # retrieve trues
    #===========================================================================
    with open(true_fp, 'rb') as f:
        true_stats_d = pickle.load(f)
        
    #===========================================================================
    # compare
    #===========================================================================
    compare_dicts(test_stats_d, true_stats_d)

 