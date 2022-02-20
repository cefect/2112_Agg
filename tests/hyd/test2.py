'''
Created on Feb. 20, 2022

@author: cefect
'''

write=False
if write:
    print('WARNING!!! runnig in write mode')


import os, copy
import pytest
import tempfile

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import numpy as np
np.random.seed(100)
from numpy.testing import assert_equal

from qgis.core import QgsVectorLayer, QgsWkbTypes


from agg.hyd.scripts import Model as CalcSession
from agg.hyd.scripts import StudyArea as CalcStudyArea
from agg.hyd.scripts import vlay_get_fdf

#===============================================================================
# fixture-----
#===============================================================================
base_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\compiled'
assert os.path.exists(base_dir)

@pytest.fixture
def session(tmp_path,
            #wrk_base_dir=None,
            wrk_base_dir=base_dir,
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
                         'wd_dir': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd',
                         #'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                            }, 
                        },
                    ):
    
    #get working directory
    wrk_dir = None
    if write:
        wrk_dir = os.path.join(wrk_base_dir, os.path.basename(tmp_path))
    
    with CalcSession(out_dir = tmp_path, proj_lib=proj_lib, wrk_dir=wrk_dir, overwrite=write,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        yield ses

@pytest.fixture
def studyAreaWrkr(session, request):
    
    name = request.param
        
    kwargs = {k:getattr(session, k) for k in ['tag', 'prec', 'trim', 'out_dir', 'overwrite']}
    with CalcStudyArea(session=session, name=name, **session.proj_lib[name]) as sa:
        yield sa

 


            
    
    
#===============================================================================
# tests------
#===============================================================================
 
@pytest.mark.parametrize('aggLevel',[10, 50], indirect=False)  
@pytest.mark.parametrize('studyAreaWrkr',['testSet1'], indirect=True)     
def test_finv_gridPoly(studyAreaWrkr, aggLevel):
    #NOTE: this function is also tested in test_finv_agg
    finv_vlay = studyAreaWrkr.get_finv_clean()
    df, finv_agg_vlay = studyAreaWrkr.get_finv_gridPoly(aggLevel=aggLevel, finv_vlay=finv_vlay)
     
    assert isinstance(finv_agg_vlay, QgsVectorLayer)
    assert isinstance(df, pd.DataFrame)
     
     
    assert finv_vlay.dataProvider().featureCount() == len(df)
    assert finv_agg_vlay.dataProvider().featureCount() <= len(df)
     
    assert 'Polygon' in QgsWkbTypes().displayString(finv_agg_vlay.wkbType())
    

    


 
@pytest.mark.parametrize('aggType,aggLevel',[['none',None], ['gridded',20], ['gridded',50]], indirect=False) 
def test_finv_agg(session, aggType, aggLevel, tmp_path):
    #===========================================================================
    # #execute the functions to be tested
    #===========================================================================
    test_d = dict()
    dkey1 = 'finv_agg_d'
    test_d[dkey1] = session.build_finv_agg(dkey=dkey1, aggType=aggType, aggLevel=aggLevel, write=write)
    
    dkey2 = 'finv_agg_mindex'
    test_d[dkey2] =session.data_d[dkey2]
    
 
    
    for dkey, test in test_d.items():
        #=======================================================================
        # get the pickle corresponding to this test
        #=======================================================================
        true_fp = search_fp(os.path.join(base_dir, os.path.basename(tmp_path)), '.pickle', dkey) #find the data file.
        assert os.path.exists(true_fp), 'failed to find match for %s'%dkey
        true = retrieve_data(dkey, true_fp, session)
        

        #=======================================================================
        # compare
        #=======================================================================
        assert len(test)==len(true)
        assert type(test)==type(true)
        
        if dkey=='finv_agg_d':
            check_layer_d(test, true, test_data=False)
 
                
        elif dkey=='finv_agg_mindex':
            assert_frame_equal(test.to_frame(), true.to_frame())
 
@pytest.mark.parametrize('tval_type',['uniform', 'rand'], indirect=False)
@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
def test_tvals(session,tval_type, finv_agg_fn, tmp_path):
    #===========================================================================
    # load inputs   
    #===========================================================================
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, session)
    
    #===========================================================================
    # execute
    #===========================================================================
    dkey='tvals'
    finv_agg_serx = session.build_tvals(dkey=dkey, tval_type=tval_type, 
                            finv_agg_d=finv_agg_d, mindex =finv_agg_mindex)
    
    #===========================================================================
    # retrieve true
    #===========================================================================
    true_fp = search_fp(os.path.join(base_dir, os.path.basename(tmp_path)), '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_series_equal(finv_agg_serx, true)
    
@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
@pytest.mark.parametrize('sgType',['centroids', 'poly'], indirect=False)  
def test_sampGeo(session, sgType, finv_agg_fn, tmp_path):
    #===========================================================================
    # load inputs   
    #===========================================================================
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, session)
        
    #===========================================================================
    # execute
    #===========================================================================
    dkey='finv_sg_d'
    vlay_d = session.build_sampGeo(dkey = dkey, sgType=sgType, finv_agg_d=finv_agg_d, write=write)
    
    #===========================================================================
    # retrieve trues    
    #===========================================================================
    
    true_fp = search_fp(os.path.join(base_dir, os.path.basename(tmp_path)), '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # check
    #===========================================================================
    check_layer_d(vlay_d, true)

        
#===============================================================================
# helpers-----------
#===============================================================================
def retrieve_finv_d(finv_agg_fn, session):
    d= dict()
    for dkey in ['finv_agg_d', 'finv_agg_mindex']:
 
        input_fp = search_fp(os.path.join(base_dir, finv_agg_fn), '.pickle', dkey) #find the data file.
        assert os.path.exists(input_fp), 'failed to find match for %s'%finv_agg_fn
        d[dkey] = retrieve_data(dkey, input_fp, session)
        
    return d.values()

def retrieve_data(dkey, fp, ses):
    assert dkey in ses.data_retrieve_hndls
    hndl_d = ses.data_retrieve_hndls[dkey]
    
    return hndl_d['compiled'](fp=fp, dkey=dkey)
    

def search_fp(dirpath, ext, pattern): #get a matching file with extension and beginning
    fns = [e for e in os.listdir(dirpath) if e.endswith(ext)]
    
    result= None
    for fn in fns:
        if fn.startswith(pattern):
            result = os.path.join(dirpath, fn)
            break
        
    return result

def check_layer_d(d1, d2,
                   test_data=True,):
    
    assert d1.keys()==d2.keys()
    
    for k, vtest in d1.items():
        vtrue = d2[k]
        
        dptest, dptrue = vtest.dataProvider(), vtrue.dataProvider()
        
        assert type(vtest)==type(vtrue)
        
        #=======================================================================
        # vectorlayer checks
        #=======================================================================
        if isinstance(vtest, QgsVectorLayer):
            assert dptest.featureCount()==dptrue.featureCount()
            assert vtest.wkbType() == dptrue.wkbType()
            
            #data checks
            if test_data:
                true_df, test_df = vlay_get_fdf(vtrue).reset_index(drop=True), vlay_get_fdf(vtest).reset_index(drop=True)
                
                assert_frame_equal(true_df, test_df)
 
            
            
            
    
