'''
Created on Apr. 12, 2022

@author: cefect

integration test for hyd.model
'''
import os, tempfile, datetime, shutil
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
#===============================================================================
# QGJIS imports
#===============================================================================
from qgis.core import QgsRectangle, QgsVectorLayer, QgsApplication, QgsProject, QgsCoordinateReferenceSystem, \
    QgsProcessingFeedback, QgsProcessingContext, QgsRasterLayer


#initilize qgis
QgsApplication.setPrefixPath(r'C:/OSGeo4W/apps/qgis-ltr', True)
app = QgsApplication([], False)

app.initQgis()
qproj = QgsProject.instance()
crs = QgsCoordinateReferenceSystem('EPSG:2955')
qproj.setCrs(crs)

#init algos
feedback = QgsProcessingFeedback()
context=QgsProcessingContext()
import processing  
from processing.core.Processing import Processing
from qgis.analysis import QgsNativeAlgorithms

Processing.initialize()  
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
#===============================================================================
# customs
#===============================================================================
from agg.hyd.hrunr import run, run_autoPars
from  tests.conftest import check_layer_d
#===============================================================================
# test data parameters
#===============================================================================
base_resolution=5 #for generating test rasters
extent_fp = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0218_extent.geojson'

 


#===============================================================================
# fixtures-----
#===============================================================================
 

@pytest.fixture(scope='module')
def finv_rand(tmpdir_factory, count=50, clusters=5):
    np.random.seed(100)
    out_dir = tmpdir_factory.mktemp('finv')
    #random points inside the extent
    pts_vlay = get_randompoints(count=count,
        #output=os.path.join(out_dir, 'random_pts_%i.gpkg'%count),
        )
    
    #cluster the points
    pts_clustered_vlay = get_kmeansclustering(pts_vlay, clusters=clusters)
    
    #convert clusters to polygons
    cvh_vlay = get_minimumboundinggeometry(pts_clustered_vlay, fieldName='CLUSTER_ID',
                                         output=os.path.join(out_dir, 'cvh_rand_polys_%i.gpkg'%clusters),
                                         )
    assert os.path.exists(cvh_vlay)
    return cvh_vlay

@pytest.fixture(scope='module')
def wse_rand():
    np.random.seed(100)
    return get_randomuniformraster(base_resolution, bounds=(5,7), extent=extent_vlay.extent(), layname='wse_rand')
            
 
@pytest.fixture(scope='module')
def dem_rand():
    np.random.seed(100)
    return get_randomuniformraster(base_resolution, bounds=(8,8), extent=extent_vlay.extent(), layname='dem_rand')

@pytest.fixture(scope='module')
def proj_lib(wse_rand, dem_rand, finv_rand): #assemble a proj_lib
    """everything should be a filepath
    as these are all session scoped"""
    
    return {'testRand':{
        'EPSG':2955, 
        'finv_fp':finv_rand, 
        'aoi':extent_fp,
        'wse_fp_d':{'hi':wse_rand},
        'dem_fp_d':{base_resolution:dem_rand},
        }}
         
    
 

#===============================================================================
# tests----------
#===============================================================================

#test the proj_lib



#test the main runr
def test_01runr(proj_lib, write, tmp_path, logger, feedback, base_dir, session):
    
    write=True
    wrk_dir = None
    if write:
        wrk_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    
    #execute
    data_d, ofp_d = run(tag='test_runr', name='test', write=write, proj_lib=proj_lib,
        iters=3, 
        out_dir=tmp_path, logger=logger, feedback=feedback, wrk_dir=wrk_dir, 
        write_lib=False, write_summary=False, exit_summary=False, #dont write any summaries
        )
    
    #===========================================================================
    # compare each dkey
    #===========================================================================
    session.compiled_fp_d = ofp_d
    
    ofp_d.keys()
 
    for dkey, compiled_fp in ofp_d.items():
        #retrieve
        testData = data_d[dkey]
        trueData = session.retrieve(dkey)
        
        #compare
        if dkey in ['finv_agg_d', 'drlay_d', 'finv_sg_d']:
            check_layer_d(testData, trueData, msg=dkey)
        elif dkey in ['finv_agg_mindex']:
            assert_frame_equal(testData.to_frame(), trueData.to_frame())
        elif dkey in ['tvals_raw', 'tvals', 'rsamps', 'rloss', 'tloss']:
            if isinstance(testData, pd.Series):
                assert_series_equal(testData, trueData)
            elif isinstance(testData, pd.DataFrame):
                assert_frame_equal(testData, trueData)
            else:
                raise TypeError('unexpected type on %s: %s'%(dkey, type(testData)))
        else:
            raise IOError(dkey)
            
 


#test all runs configured in the modelPars.xls


#===============================================================================
# helpers
#===============================================================================
def vlay_load(fp):
    """TODO: move this to conftest"""
    if fp is None: fp = extent_fp
    assert os.path.exists(fp)
    fname, ext = os.path.splitext(os.path.split(fp)[1])
    vlay_raw = QgsVectorLayer(fp,fname,'ogr') #not sure why geojson isnt working
    
    assert vlay_raw.isValid()
    
    assert vlay_raw.crs() == qproj.crs()
    
    return vlay_raw

extent_vlay = vlay_load(extent_fp)

def rlay_load(fp, layname=None):
    """TODO: move this to conftest"""
    if layname is None: layname=os.path.splitext(os.path.split(fp)[1])[0]
    rlay_raw = QgsRasterLayer(fp, layname)
    
    assert isinstance(rlay_raw, QgsRasterLayer), 'failed to get a QgsRasterLayer'
    assert rlay_raw.isValid(), "Layer failed to load!"
    assert rlay_raw.crs() == qproj.crs()
    
    return rlay_raw

def get_randomuniformraster(
            pixel_size,
            bounds=(0,1),
            extent=None, #layer to pull raster extents from
 
            output='TEMPORARY_OUTPUT',
            layname=None,
 
        ):
    if layname is None: layname='randomuniformraster'

    ins_d = { 'EXTENT' : extent,
                  'LOWER_BOUND' : bounds[0], 'UPPER_BOUND' : bounds[1],
                  'OUTPUT' : output, 
                 'OUTPUT_TYPE' : 5, #Float32
                 'PIXEL_SIZE' : pixel_size, 
                 'TARGET_CRS' :qproj.crs(), 
                 }
    algo_nm = 'native:createrandomuniformrasterlayer'
    ofp =  processing.run(algo_nm, ins_d,  feedback=feedback, context=context)['OUTPUT']
    
    return ofp
    #return rlay_load(ofp, layname=layname)
    
    
def get_randompoints(
        count=10,
        min_distance=10,
        output='TEMPORARY_OUTPUT',
        ):
    ins_d = { 'EXTENT' : extent_vlay.extent(),
              'MAX_ATTEMPTS' : 200, 'MIN_DISTANCE' : min_distance,
               'OUTPUT' : output, 
               'POINTS_NUMBER' : count, 'TARGET_CRS' : qproj.crs() }
    
    
    algo_nm = 'native:randompointsinextent'
    ofp =  processing.run(algo_nm, ins_d,  feedback=feedback, context=context)['OUTPUT']
    
    return ofp

def get_kmeansclustering(
        vlay, clusters, fieldName='CLUSTER_ID',
        output='TEMPORARY_OUTPUT',
        ):
    
        ins_d = { 'CLUSTERS' : clusters, 
                 'FIELD_NAME' : fieldName, 
                 'SIZE_FIELD_NAME' : 'CLUSTER_SIZE',
                 'INPUT' : vlay, 
                 'OUTPUT' : output,}
        
        algo_nm = 'native:kmeansclustering'
        
        return processing.run(algo_nm, ins_d,  feedback=feedback, context=context)['OUTPUT']
    
def get_minimumboundinggeometry(
                     vlay,
                     fieldName=None, #optional category name
                     output='TEMPORARY_OUTPUT',
        ):
    algo_nm = 'qgis:minimumboundinggeometry'
    ins_d = { 'FIELD' : fieldName, 
                 'INPUT' : vlay, 
                 'OUTPUT' : output, 
                 'TYPE' : 3, #convex hull
                  }
    return processing.run(algo_nm, ins_d,  feedback=feedback, context=context)['OUTPUT']
            
    