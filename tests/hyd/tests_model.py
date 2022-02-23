'''
Created on Feb. 20, 2022

@author: cefect

tests for hyd model
'''




import os  
import pytest
 

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
idx = pd.IndexSlice

import numpy as np
np.random.seed(100)
from numpy.testing import assert_equal

from qgis.core import QgsVectorLayer, QgsWkbTypes


from agg.hyd.scripts import Model as CalcSession
from agg.hyd.scripts import StudyArea as CalcStudyArea
from agg.hyd.scripts import vlay_get_fdf

from tests.conftest import retrieve_finv_d, retrieve_data, search_fp
#===============================================================================
# fixtures-----
#===============================================================================

@pytest.fixture(scope='module')
def df_d():
    """this is an expensive collecting of csvs (database dump) used to build vfuncs
    keeping this alive for the whole module"""
    with CalcSession() as ses:
        df_d = ses.build_df_d(dkey='df_d')
    return df_d



@pytest.fixture
def session(tmp_path,
            #wrk_base_dir=None, 
            base_dir, write,logger, feedback,#see conftest.py (scope=session)
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
    
    with CalcSession(out_dir = tmp_path, proj_lib=proj_lib, wrk_dir=wrk_dir, 
                     overwrite=write,write=write, logger=logger,feedback=feedback,
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
def test_finv_agg(session, aggType, aggLevel, true_dir, base_dir, write):
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
        true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
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

@pytest.mark.dev
@pytest.mark.parametrize('tval_type',['rand'], indirect=False)
@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
def test_tvals(session,tval_type, finv_agg_fn, true_dir, base_dir, write):
    #===========================================================================
    # load inputs   
    #===========================================================================
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, session, base_dir)
    
    #===========================================================================
    # execute
    #===========================================================================
    dkey='tvals'
    finv_agg_serx = session.build_tvals(dkey=dkey, tval_type=tval_type, 
                            finv_agg_d=finv_agg_d, mindex =finv_agg_mindex, write=write)
    
    #===========================================================================
    # retrieve true
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_series_equal(finv_agg_serx, true)
    
@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
@pytest.mark.parametrize('sgType',['centroids', 'poly'], indirect=False)  
def test_sampGeo(session, sgType, finv_agg_fn, true_dir, write, base_dir):
    #===========================================================================
    # load inputs   
    #===========================================================================
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, session, base_dir)
        
    #===========================================================================
    # execute
    #===========================================================================
    dkey='finv_sg_d'
    vlay_d = session.build_sampGeo(dkey = dkey, sgType=sgType, finv_agg_d=finv_agg_d, write=write)
    
    #===========================================================================
    # retrieve trues    
    #===========================================================================
    
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # check
    #===========================================================================
    check_layer_d(vlay_d, true)

#@pytest.mark.parametrize('finv_sg_d_fn',['test_sampGeo_centroids_test_fi1', 'test_sampGeo_poly_test_finv_ag1'], indirect=False)
#rsamps methods are only applicable for certain geometry types  

@pytest.mark.parametrize('finv_sg_d_fn',[ #see test_sampGeo
    'test_sampGeo_poly_test_finv_ag0','test_sampGeo_poly_test_finv_ag1',])
@pytest.mark.parametrize('samp_method',['zonal'], indirect=False)
@pytest.mark.parametrize('zonal_stat',['Mean','Minimum', 'Maximum'])  
def test_rsamps_poly(session, finv_sg_d_fn,samp_method, true_dir, write, base_dir, zonal_stat):
 
    rsamps_runr(base_dir, true_dir, session, zonal_stat=zonal_stat,
                samp_method=samp_method, write=write, finv_sg_d_fn=finv_sg_d_fn)
    


@pytest.mark.parametrize('finv_sg_d_fn',[ #see test_sampGeo
    'test_sampGeo_centroids_test_fi1','test_sampGeo_centroids_test_fi0'])
@pytest.mark.parametrize('samp_method',['points'], indirect=False) 
def test_rsamps_point(session, finv_sg_d_fn,samp_method, true_dir, write, base_dir):
 
    rsamps_runr(base_dir, true_dir,session, 
                samp_method=samp_method, write=write, finv_sg_d_fn=finv_sg_d_fn)
    



@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
@pytest.mark.parametrize('sgType',['centroids', 'poly'], indirect=False)  
@pytest.mark.parametrize('samp_method',['true_mean'], indirect=False) 
def test_rsamps_trueMean(session, finv_agg_fn, samp_method, true_dir, write, base_dir, sgType):
    #===========================================================================
    # build the sample geometry
    #===========================================================================
    """because true_mean requires the raw inventory.. 
        for this test we perform the previous calc (sample geometry) as well
        to simplify the inputs"""
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, session, base_dir)
    
    finv_sg_d = session.build_sampGeo(dkey = 'finv_sg_d', sgType=sgType, finv_agg_d=finv_agg_d, write=False)
 
    #===========================================================================
    # execute
    #===========================================================================        
 
    rsamps_runr(base_dir, true_dir,session, samp_method=samp_method, write=write, 
                finv_sg_d=finv_sg_d, mindex=finv_agg_mindex)

def rsamps_runr(base_dir, true_dir,session,finv_sg_d=None,finv_sg_d_fn=None, **kwargs):
    """because kwarg combinations are complex for rsamps... its easier to split out the tests"""
    #===========================================================================
    # load inputs   
    #===========================================================================
    if finv_sg_d is None:
        dkey = 'finv_sg_d'
        input_fp = search_fp(os.path.join(base_dir, finv_sg_d_fn), '.pickle', dkey) #find the data file.
        finv_sg_d = retrieve_data(dkey, input_fp, session)
 
    #===========================================================================
    # execute
    #===========================================================================

    dkey='rsamps'
    rsamps_serx = session.build_rsamps(dkey=dkey, finv_sg_d=finv_sg_d, **kwargs)
    
    #===========================================================================
    # basic checks
    #===========================================================================
 

    
    #===========================================================================
    # retrieve trues    
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_series_equal(rsamps_serx, true)


@pytest.mark.parametrize('rsamp_fn', #see test_rsamps
             ['test_rsamps_test_finv_agg_grid0', 'test_rsamps_test_finv_agg_grid1', 'test_rsamps_test_finv_agg_grid2']) 
@pytest.mark.parametrize('vid', [49, 798,811, 0])
def test_rloss(session, rsamp_fn, vid, base_dir, true_dir, df_d):
 
    #===========================================================================
    # load inputs
    #===========================================================================
    dkey = 'rsamps'
    input_fp = search_fp(os.path.join(base_dir, rsamp_fn), '.pickle', dkey) #find the data file.
    dxser = retrieve_data(dkey, input_fp, session)
    
    #===========================================================================
    # execute
    #===========================================================================
    dkey='rloss'
    rdxind = session.build_rloss(dkey=dkey, vid=vid, dxser=dxser, df_d=df_d)
    
    #===========================================================================
    # check
    #===========================================================================
    rserx = rdxind['rl']
    assert rserx.notna().all()
    assert rserx.min()>=0
    assert rserx.max()<=100
    
    #===========================================================================
    # retrieve trues
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_frame_equal(rdxind, true)

rloss_fn_l = ['test_rloss_49_test_rsamps_test0','test_rloss_49_test_rsamps_test1','test_rloss_49_test_rsamps_test2',
              'test_rloss_798_test_rsamps_tes0','test_rloss_798_test_rsamps_tes1','test_rloss_798_test_rsamps_tes2',
              'test_rloss_811_test_rsamps_tes0','test_rloss_811_test_rsamps_tes1','test_rloss_811_test_rsamps_tes2']
 
@pytest.mark.parametrize('rloss_fn', rloss_fn_l) #see test_rloss
def test_tloss(session, base_dir, rloss_fn):
    #scale_cn = session.scale_cn
    #===========================================================================
    # load inputs
    #===========================================================================
    dkey = 'rloss'
    input_fp = search_fp(os.path.join(base_dir, rloss_fn), '.pickle', dkey) #find the data file.
    rl_dxind = retrieve_data(dkey, input_fp, session)
    
    #build total vals
    """easier (and broader) to use random total vals than to select the matching"""
    
    tv_dx1 = pd.Series(np.random.random(len(rl_dxind)), index=rl_dxind.droplevel(1).index, name='0').to_frame()
    tv_dx2 = pd.concat({'tvals':tv_dx1}, axis=1, names=['dkey', 'iter'])
    #===========================================================================
    # execute
    #===========================================================================
    dkey='tloss'
    tl_dxind = session.build_tloss(dkey=dkey, tv_data=tv_dx2, rl_dxind=rl_dxind)
    
    #===========================================================================
    # check
    #===========================================================================
    assert_frame_equal(tl_dxind.loc[:, idx['tvals', :]].droplevel(1, axis=0), tv_dx2, check_index_type=False)
    
    #check relative loss
    rl_dx_chk = tl_dxind.loc[idx[:, 'tloss']].droplevel(1, axis=0).divide(tv_dx2.loc[idx[:, 'tvals']])
    rl_dx_chk.columns = pd.Index(range(len(rl_dx_chk.columns)))
    
    rl_dx_chk2 = tl_dxind.loc[idx[:, 'rloss']].droplevel(1, axis=0)
    rl_dx_chk2.columns = pd.Index(range(len(rl_dx_chk2.columns)))
    assert_frame_equal(rl_dx_chk, rl_dx_chk2)
 
 

    
    
        
#===============================================================================
# helpers-----------
#===============================================================================

    



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
 
            
            
            
    
