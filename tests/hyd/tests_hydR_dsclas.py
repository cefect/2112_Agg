'''
Created on Aug. 20, 2022

@author: cefect

unit tests for downsample classification
'''
from qgis.core import QgsCoordinateReferenceSystem
import pytest, copy, os, random
import numpy as np
import pandas as pd

 
import rasterio as rio

from tests.conftest import validate_raster, validate_dict, src_dir, get_abs
from hp.rio import RioWrkr, write_array, load_array
from hp.np import apply_blockwise_ufunc, apply_blockwise
 

#scripts to test
from agg.hydR.dsclas import DsampClassifier, get_wse_filtered

#===============================================================================
# test data
#===============================================================================
#toy example used in figure
toy_dem_ar = np.array((
                [7,10,4,4],
                [6,8,3,1],
                [8,5,4,8],
                [1,3,8,9],
                #[1,1,1,1],
                #[1,1,1,1]
                ), dtype=np.float64)


toy_wse_ar = get_wse_filtered(np.full((4,4), 5, dtype=np.float64), toy_dem_ar)

rand_dem_ar = np.random.random((4,6))*10
rand_wse_ar =  get_wse_filtered(np.random.random(rand_dem_ar.shape)*10, rand_dem_ar)
#===============================================================================
# helpers and globals------
#===============================================================================
crs = QgsCoordinateReferenceSystem('EPSG:2953')
prec=5

def get_rand_wse_ar(dem_ar):
    """build a noise wse layer from a dem_ar"""
    dem_ar
    

 

def rlay_to_file(ar, ofp):
    
    return write_array(ar, ofp,
                crs=rio.crs.CRS.from_epsg(2953),
                transform=rio.transform.from_origin(2476176,7447040,1,1), 
                )
    

#===============================================================================
# FIXTURES-----
#===============================================================================

@pytest.fixture(scope='function')
def dem_fp(dem_ar, tmp_path): 
    return rlay_to_file(dem_ar, os.path.join(tmp_path, 'dem.tif'))
 

@pytest.fixture(scope='function')
def wse_fp(wse_ar, tmp_path): 
    return rlay_to_file(wse_ar, os.path.join(tmp_path, 'wse.tif'))


@pytest.fixture(scope='function')
def dscWrkr(tmp_path,write,logger, feedback,  test_name,             
            qgis_app, qgis_processing, #pytest-qgis fixtures 
                    ):
    
    """Mock session for tests"""
 
    np.random.seed(100)
    random.seed(100)
    
 
    
    with DsampClassifier( 
                #Qcoms
                 compression='none',  
                 crs=crs,
                 feedback=feedback,
                 qgis_app=qgis_app,qgis_processing=True, #pytest-qgis
                 
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
# TESTS--------
#===============================================================================
@pytest.mark.parametrize('dem_ar', [toy_dem_ar, np.random.random((10,20))*10])
@pytest.mark.parametrize('downscale',[2]) 
def test_01_demCoarse(dem_fp,   dscWrkr, downscale, dem_ar):
    
    #build with function
    test_fp = dscWrkr.build_coarse(dem_fp, downscale=downscale)
    
    #===========================================================================
    # #validate
    #===========================================================================
    test_ar = load_array(test_fp)
    
 
    #compute downscale w/ numpy
    vali_ar = apply_blockwise(dem_ar, np.mean, n=downscale)
 
    """having some issues with precision on the rasterio load"""
    assert np.array_equal(test_ar.round(2), vali_ar.round(2))
    

 
@pytest.mark.parametrize('dem_ar, wse_ar', [
    (toy_dem_ar, toy_wse_ar),
    #(np.random.random((10,20))*10
    ])
def test_02_fineDelta(dem_ar, dem_fp,wse_ar, wse_fp,  dscWrkr):
    
    #build with function
    test_fp = dscWrkr.build_delta(dem_fp, wse_fp)
    
    #===========================================================================
    # #validate
    #===========================================================================
    test_ar = load_array(test_fp)
    
 
    #compute w/ numpy
    vali_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
 
    """having some issues with precision on the rasterio load"""
    assert np.array_equal(test_ar.round(2), vali_ar.round(2))
    
    
@pytest.mark.dev
@pytest.mark.parametrize('dem_ar, wse_ar', [
    (toy_dem_ar, toy_wse_ar),
    #(rand_dem_ar, rand_wse_ar),
    ])
@pytest.mark.parametrize('downscale',[2]) 
def test_03_catMask(dem_ar, dem_fp,wse_ar, wse_fp,  dscWrkr, downscale):
    
    #get coarse DEM
    """not ideal as a unit test... but easier than handling something premade"""
    demC_fp = dscWrkr.build_coarse(dem_fp, downscale=downscale)
    
    #build with function
    test_fp = dscWrkr.build_cat_masks(dem_fp, demC_fp, wse_fp)
    
    #===========================================================================
    # #validate
    #===========================================================================
    test_ar = load_array(test_fp)
    
 
    #compute w/ numpy
    vali_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
 
    """having some issues with precision on the rasterio load"""
    assert np.array_equal(test_ar.round(2), vali_ar.round(2))
    
    
    
    