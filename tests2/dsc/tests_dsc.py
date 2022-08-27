'''
Created on Aug. 20, 2022

@author: cefect

unit tests for downsample classification
'''
#from qgis.core import QgsCoordinateReferenceSystem

 
#scripts to test
from agg2.haz.dsc.scripts import DsampClassifier
from agg2.haz.misc import get_rand_ar, get_wse_filtered
from hp.np import apply_blockwise_ufunc, apply_blockwise, dropna
from hp.rio import RioWrkr, write_array, load_array
from numpy import array, dtype

import numpy as np
import pytest, copy, os, random
import rasterio as rio
xfail = pytest.mark.xfail

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



#===============================================================================
# rand_dem_ar = np.random.random((4,6))*10
# rand_wse_ar =  get_wse_filtered(np.random.random(rand_dem_ar.shape)*10, rand_dem_ar)
#===============================================================================


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
def dscWrkr(tmp_path,write,logger, test_name,             
            #qgis_app, qgis_processing, feedback, #pytest-qgis fixtures 
                    ):
    
    """Mock session for tests"""
 
    np.random.seed(100)
    random.seed(100)
    
 
    
    with DsampClassifier( 
                #Qcoms
                 #==============================================================
                 # compression='none',  
                 # crs=QgsCoordinateReferenceSystem('EPSG:%i'%crsid),
                 # feedback=feedback,
                 # qgis_app=qgis_app,qgis_processing=True, #pytest-qgis
                 #==============================================================
                 
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

@pytest.mark.parametrize('dem_ar, downscale', [
    #(np.random.random((3, 4))*10, 2),
    (get_wse_filtered(np.random.random((7, 8))*10, np.random.random((7, 8))*10), 2), #wse
 
    ])
def test_00_crop(dem_fp,   dscWrkr, downscale, dem_ar):
    
    #build with function
    test_fp = dscWrkr.build_crop(dem_fp, divisor=downscale)
    
    #===========================================================================
    # #validate
    #===========================================================================
    test_ar = load_array(test_fp)
    
    for dim in test_ar.shape:
        assert dim%downscale==0
 
    assert np.all(np.isin(dropna(test_ar), dropna(dem_ar)))
    
    
    

@pytest.mark.parametrize('dem_ar, downscale', [
    (toy_dem_ar, 2),  
    (np.random.random((4*3,4*10))*10, 4),
    pytest.param(np.random.random((3, 4))*10, 2, marks=xfail(strict=True, reason='bad shape')),
    ])
 
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
    get_rand_ar((4,6))
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
    
#===============================================================================
# category masks
#===============================================================================
catMask_d = {
    'toy':{
        'DD': array([
            [ True,  True, False, False],
           [ True,  True, False, False],
           [False, False, False, False],
           [False, False, False, False]]), 
        'WW': array([
            [False, False,  True,  True],
           [False, False,  True,  True],
           [False, False, False, False],
           [False, False, False, False]]), 
        'WP': array([
            [False, False, False, False],
           [False, False, False, False],
           [ True,  True, False, False],
           [ True,  True, False, False]]), 
        'DP': array([
            [False, False, False, False],
            [False, False, False, False],
            [False, False,  True,  True],
            [False, False,  True,  True]
           ])},
    'rand':None, #too messy to have consistent outputs here
    }

@pytest.mark.dev
@pytest.mark.parametrize('dem_ar, wse_ar, vali_d', [
    (toy_dem_ar, toy_wse_ar, catMask_d['toy']),
    list(get_rand_ar((8,6)))+ [None]
    ])
@pytest.mark.parametrize('downscale',[2]) 
def test_03_catMask(dem_ar, dem_fp,wse_ar, wse_fp,  dscWrkr, downscale, vali_d):
    
    #get coarse DEM
    """not ideal as a unit test... but easier than handling something premade"""
    demC_fp = dscWrkr.build_coarse(dem_fp, downscale=downscale)
    
    #build with function
    test_d, _ = dscWrkr.build_cat_masks(dem_fp, demC_fp, wse_fp, write=False)
    
    #===========================================================================
    # #validate
    #===========================================================================
    miss_l = set(['DD', 'WW', 'WP', 'DP']).symmetric_difference(test_d.keys())
    assert len(miss_l)==0
    
    for k,v in test_d.items():
        assert isinstance(v, np.ndarray), k
        assert v.dtype==np.dtype('bool'), k
        if not vali_d is None:
            assert np.array_equal(v, vali_d[k]), k
            
    
            
    return

cmMosaic_ar = array([
        [11, 11, 21, 21],
       [11, 11, 21, 21],
       [31, 31, 41, 41],
       [31, 31, 41, 41]])


@pytest.mark.parametrize('cm_d, vali_ar',[[catMask_d['toy'], cmMosaic_ar]]) 
def test_04_cmMosaic(cm_d, vali_ar, dscWrkr):
    """build_cat_mosaic"""
    #build the array
    cm_ar, test_fp = dscWrkr.build_cat_mosaic(cm_d,write=True,output_kwargs=output_kwargs)
    
    #===========================================================================
    # validate
    #===========================================================================
    assert isinstance(cm_ar, np.ndarray)
    assert cm_ar.dtype==dtype('int32')
    assert np.array_equal(cm_ar, vali_ar)
    assert os.path.exists(test_fp)
    

#===============================================================================
# INTEGRATION TEST----
#===============================================================================

@pytest.mark.parametrize('dem_ar, wse_ar, downscale, vali_ar', [
    (toy_dem_ar, toy_wse_ar, 2, cmMosaic_ar),
    #list(get_rand_ar((4,6)))+ [2, None], 
    list(get_rand_ar((2*100,2*200)))+ [2, None], 
    #list(get_rand_ar((3,4)))+ [2, None], 
    ])
def test_05_all(dem_ar, dem_fp,wse_ar, wse_fp,  vali_ar, downscale, dscWrkr):
    
    #base the worker on the dem
    dscWrkr._base_set(dem_fp)
    dscWrkr._base_inherit()
 
    
    #run the chain to build the mosaic
    test_fp = dscWrkr.run_all(dem_fp, wse_fp, downscale=downscale, write=True)
    
    #===========================================================================
    # validate
    #===========================================================================
    cm_ar = load_array(test_fp)
    assert isinstance(cm_ar, np.ndarray)
    assert cm_ar.dtype==dtype('int32')
    
    if not vali_ar is None:
        assert np.array_equal(cm_ar, vali_ar)
    