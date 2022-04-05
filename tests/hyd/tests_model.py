'''
Created on Feb. 20, 2022

@author: cefect

tests for hyd model

TODO: clean out old pickles
'''




import os  
import pytest
 

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
idx = pd.IndexSlice

import numpy as np
np.random.seed(100)
from numpy.testing import assert_equal

from qgis.core import QgsVectorLayer, QgsWkbTypes
import hp.gdal

from agg.hyd.scripts import Model as CalcSession
from agg.hyd.scripts import StudyArea as CalcStudyArea
from agg.hyd.scripts import vlay_get_fdf, RasterCalc

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



@pytest.fixture(scope='function')
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
                         #'wd_fp':r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd\wd_rand_test_0304.tif',

                            }, 
                        },
                    ):
 
    np.random.seed(100)
    
    #get working directory
    wrk_dir = None
    if write:
        wrk_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    
    with CalcSession(out_dir = tmp_path, proj_lib=proj_lib, wrk_dir=wrk_dir, 
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        
        assert len(ses.data_d)==0
        yield ses

 
@pytest.fixture
def studyAreaWrkr(session, request):
    
    name = request.param
        
    kwargs = {k:getattr(session, k) for k in ['tag', 'prec', 'trim', 'out_dir', 'overwrite']}
    with CalcStudyArea(session=session, name=name, **session.proj_lib[name]) as sa:
        yield sa

 

@pytest.fixture   
def dem_fp(studyAreaWrkr, tmp_path):
    return studyAreaWrkr.randomuniformraster(5, bounds=(0,5), extent_layer=studyAreaWrkr.finv_vlay,
                                             output=os.path.join(tmp_path, 'dem_random.tif'))

@pytest.fixture
# were setup to filter out ground water... but tests are much simpler if we ignore this   
def wse_fp(studyAreaWrkr, tmp_path):
    return studyAreaWrkr.randomuniformraster(5, bounds=(5,7), extent_layer=studyAreaWrkr.finv_vlay,
                                             output=os.path.join(tmp_path, 'wse_random.tif'))
    
@pytest.fixture   
def wd_rlay(session):
        wd_fp = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd\wd_rand_test_0304.tif'
        rlay = session.rlay_load(wd_fp)
        session.mstore.addMapLayer(rlay)
        return rlay
    

#===============================================================================
# TESTS STUDYAREA------
#===============================================================================
 
@pytest.mark.parametrize('aggLevel',[10, 50], indirect=False)  
@pytest.mark.parametrize('studyAreaWrkr',['testSet1'], indirect=True)     
def test_finv_gridPoly(studyAreaWrkr, aggLevel):
    """"this function is also tested in test_finv_agg"""
    finv_vlay = studyAreaWrkr.get_finv_clean()
    df, finv_agg_vlay = studyAreaWrkr.get_finv_gridPoly(aggLevel=aggLevel, finv_vlay=finv_vlay)
     
    assert isinstance(finv_agg_vlay, QgsVectorLayer)
    assert isinstance(df, pd.DataFrame)
     
     
    assert finv_vlay.dataProvider().featureCount() == len(df)
    assert finv_agg_vlay.dataProvider().featureCount() <= len(df)
     
    assert 'Polygon' in QgsWkbTypes().displayString(finv_agg_vlay.wkbType())

@pytest.mark.dev
@pytest.mark.parametrize('studyAreaWrkr',['testSet1'], indirect=True) 
@pytest.mark.parametrize('resampStage, resolution, resampling',[
    ['none',5, 'none'], #raw... no rexampling
    ['depth',50,'Average'],
    ['wse',50,'Average'],
    ['depth',50,'Maximum'],
    ])  
def test_get_drlay(studyAreaWrkr, resampStage, resolution, resampling, 
                   dem_fp, wse_fp, #randomly generated rasters 
                   tmp_path):
    
    #===========================================================================
    # get calc result
    #===========================================================================
    rlay = studyAreaWrkr.get_drlay(
        wse_fp_d = {'hi':wse_fp},
        dem_fp_d = {5:dem_fp},
        resolution=resolution, resampling=resampling, resampStage=resampStage)
    
    #===========================================================================
    # check result----
    #===========================================================================
    #resulting stats
    stats_d = studyAreaWrkr.rlay_getstats(rlay)
    assert stats_d['resolution']==resolution
        
    #check nodata values
    assert hp.gdal.getNoDataCount(rlay.source())==0
    assert rlay.crs() == studyAreaWrkr.qproj.crs()
    #stats_d = studyAreaWrkr.rlay_getstats(rlay)
    
    #===========================================================================
    # against the raw depth
    #===========================================================================
    """what is the resolution of the test data??"""
    with RasterCalc(wse_fp, session=studyAreaWrkr, out_dir=tmp_path, logger=studyAreaWrkr.logger) as wrkr:
        #dep_rlay = wrkr.ref_lay
        wse_rlay = wrkr.ref_lay #loaded during init
        dtm_rlay = wrkr.load(dem_fp)
 
 
        entries_d = {k:wrkr._rCalcEntry(v) for k,v in {'wse':wse_rlay, 'dtm':dtm_rlay}.items()}
        
        formula = '{wse} - {dtm}'.format(**{k:v.ref for k,v in entries_d.items()})
        
        chk_rlay_fp = wrkr.rcalc(formula, report=False)
        
        #full resolution test calc
        stats2_d = studyAreaWrkr.rlay_getstats(chk_rlay_fp)
        

        
        assert stats2_d['resolution']<=stats_d['resolution']
        
        if resampling =='Average':
            assert abs(stats2_d['MEAN'] - stats_d['MEAN']) <1.0
            assert stats2_d['MAX'] >=stats_d['MAX']
            assert stats2_d['MIN'] <=stats_d['MIN']
            assert stats2_d['RANGE']>=stats_d['RANGE']
            
        if resampling =='Maximum':
            assert abs(stats2_d['MAX'] - stats_d['MAX']) <0.001
            
            
        
        if resolution==0:
            for stat, val in {k:stats_d[k] for k in ['MAX', 'MIN']}.items():
                assert abs(val)<1e-3, stat
            
 
    
    
        
 
    
#===============================================================================
# tests Session-------
#===============================================================================
    

@pytest.mark.parametrize('aggType,aggLevel',[['none',0], ['gridded',20], ['gridded',50]], indirect=False) 
def test_finv_agg(session, aggType, aggLevel, true_dir, write):
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


@pytest.mark.parametrize('tval_type',['uniform'], indirect=False) #rand is silly here. see test_stoch also
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('finv_agg_fn, dscale_meth',[
                                        ['test_finv_agg_gridded_50_0', 'centroid'], 
                                        ['test_finv_agg_none_None_0', 'none'],
                                        ], indirect=False)  #see test_finv_agg
def test_tvals(session,finv_agg_fn, true_dir, base_dir, write, tval_type, normed, dscale_meth):
    norm_scale=1.0
    dkey='tvals'
 
    #===========================================================================
    # load inputs   
    #===========================================================================
 
    
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, session, base_dir)
    
    #===========================================================================
    # execute
    #===========================================================================
    
    finv_agg_serx = session.build_tvals(dkey=dkey, norm_scale=norm_scale,
                                    tval_type=tval_type, normed=normed, dscale_meth=dscale_meth,
                            #finv_agg_d=finv_agg_d,
                             mindex =finv_agg_mindex, write=write)
    
    #data checks
    assert_index_equal(finv_agg_mindex.droplevel('id').drop_duplicates(), finv_agg_serx.index)
    
 
    if normed:
        assert (finv_agg_serx.groupby(level='studyArea').sum().round(3)==norm_scale).all()
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




#===============================================================================
# Rsamp tests
#===============================================================================
#rsamps methods are only applicable for certain geometry types  
 
@pytest.mark.parametrize('finv_sg_d_fn',[ #see test_sampGeo
    'test_sampGeo_poly_test_finv_ag0','test_sampGeo_poly_test_finv_ag1',])
@pytest.mark.parametrize('samp_method',['zonal'], indirect=False)
@pytest.mark.parametrize('zonal_stat',['Mean','Minimum', 'Maximum'])  

def test_rsamps_poly(session, finv_sg_d_fn,samp_method, true_dir, write, base_dir, zonal_stat,wd_rlay,
                     ):
 
    rsamps_runr(base_dir, true_dir, session, zonal_stat=zonal_stat,
                samp_method=samp_method, write=write, finv_sg_d_fn=finv_sg_d_fn,wd_rlay=wd_rlay,
                )
    

    


@pytest.mark.parametrize('finv_sg_d_fn',[ #see test_sampGeo
    'test_sampGeo_centroids_test_fi1','test_sampGeo_centroids_test_fi0'])
@pytest.mark.parametrize('samp_method',['points'], indirect=False) 
def test_rsamps_point(session, finv_sg_d_fn,samp_method, true_dir, write, base_dir, wd_rlay):
 
    rsamps_runr(base_dir, true_dir,session, wd_rlay=wd_rlay,
                samp_method=samp_method, write=write, finv_sg_d_fn=finv_sg_d_fn)
    
    
    



@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
@pytest.mark.parametrize('sgType',['centroids', 'poly'], indirect=False)  
@pytest.mark.parametrize('samp_method',['true_mean'], indirect=False) 
def test_rsamps_trueMean(session, finv_agg_fn, samp_method, true_dir, write, base_dir, sgType, wd_rlay):
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
                finv_sg_d=finv_sg_d, mindex=finv_agg_mindex, wd_rlay=wd_rlay)

def rsamps_runr(base_dir, true_dir,session,finv_sg_d=None,finv_sg_d_fn=None, wd_rlay=wd_rlay, **kwargs):
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
    saName = list(session.proj_lib.keys())[0]
    
    dkey='rsamps'
    rsamps_serx = session.build_rsamps(dkey=dkey, finv_sg_d=finv_sg_d, 
                                       drlay_d={saName:wd_rlay},
                                        **kwargs)
    
 
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
def check_layer_d(d1, d2, #two containers of layers
                   test_data=True, #check vlay attributes
                   ignore_fid=True,  #whether to ignore the native ordering of the vlay
                   ):
    
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
                true_df, test_df = vlay_get_fdf(vtrue), vlay_get_fdf(vtest)
                
                if ignore_fid:
                    true_df = true_df.sort_values(true_df.columns[0],  ignore_index=True) #sort by first column and reset index
                    test_df = test_df.sort_values(test_df.columns[0],  ignore_index=True)
                
                
                assert_frame_equal(true_df, test_df,check_names=False)
 
            
            
            
    
