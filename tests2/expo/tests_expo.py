'''
Created on Sep. 6, 2022

@author: cefect
'''

from tests2.conftest import proj_d
import numpy as np
import pandas as pd
import pytest, copy, os, random
from agg2.expo.scripts import ExpoSession as Session
from agg2.expo.run import run_expo
from agg2.haz.coms import cm_int_d
from hp.rio import write_array
import shapely.geometry as sgeo
import rasterio as rio

from tests2.conftest import bbox_base, crs

#===============================================================================
# vars
#===============================================================================

bbox1 = sgeo.box(25, 25, 70, 70)

#===============================================================================
# HELPERS--------
#===============================================================================
 
#===============================================================================
# FIXTURES-----
#===============================================================================



@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name, 
                    ):
    
    """Mock session for tests"""
 
    #np.random.seed(100)
    #random.seed(100)
 
    
    with Session(  
                 #GeoPandas
                 crs=crs,
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

#===============================================================================
# ressampl class mask
#===============================================================================

 


@pytest.fixture(scope='function')
def cMask_pick_fp(cMask_rlay_fp, tmp_path):
    """mimic output of run_catMasks"""
    df = pd.DataFrame.from_dict(
        {'downscale':[1,2],'catMosaic':[np.nan, cMask_rlay_fp],}
        )
    ofp = os.path.join(tmp_path, 'test_cMasks_%i.pkl'%len(df))
    df.to_pickle(ofp)
    
    return ofp
 

@pytest.fixture(scope='function')
def cMask_rlay_fp(cMask_ar, tmp_path):
    ofp = os.path.join(tmp_path, 'cMask_%i.tif'%cMask_ar.size)
    
    width, height = cMask_ar.shape
    
    write_array(cMask_ar, ofp, crs=crs,
                 transform=rio.transform.from_bounds(*bbox_base.bounds,width, height),  
                 masked=False)
    
    return ofp
 
    
 
@pytest.fixture(scope='function')    
def cMask_ar(shape):
    return np.random.choice(np.array(list(cm_int_d.values())), size=shape)

#===============================================================================
# data grids
#===============================================================================
@pytest.fixture(scope='function')  
def complete_pick_fp(tmp_path, dsc_l, shape):
    """construct teh complete pickle of each layers stack
    equivalent to the expectations for 'catMasks'"""
    
    #build the resolution-filepath stack for each layer
    d = dict()
    for layName in ['wse', 'wd', 'catMosaic']:
        ar_d = get_ar_d(shape, dsc_l, layName)
        d[layName] = get_rlay_fp_d(ar_d, layName, tmp_path)
        
    
    df = pd.DataFrame.from_dict(d).rename_axis('downscale')
    
    #clear out the first catMask
    df.loc[dsc_l[0], 'catMosaic'] = np.nan
    
    #move downscale to a column
    df = df.reset_index()
    
 
    
 
    ofp = os.path.join(tmp_path, 'test_complete_%i.pkl'%len(df))
    df.to_pickle(ofp)
    
    return ofp
    
@pytest.fixture(scope='function')
@pytest.mark.parametrize('layName', ['wd'])
def lay_pick_fp_wd(lay_pick_fp):
    return lay_pick_fp
    

@pytest.fixture(scope='function')
def lay_pick_fp(rlay_fp_d, layName, tmp_path):
    return get_lay_pick_fp(rlay_fp_d, layName, tmp_path)
    
def get_lay_pick_fp(rlay_fp_d, layName, tmp_path):
    
    df = pd.Series(rlay_fp_d).rename(layName).to_frame().rename_axis('downscale').reset_index()
 
    ofp = os.path.join(tmp_path, 't%s_%i.pkl'%(layName, len(df)))
    df.to_pickle(ofp)
    
    return ofp


@pytest.fixture(scope='function')
def rlay_fp_d(ar_d, layName, tmp_path):
    return get_rlay_fp_d(ar_d, layName, tmp_path)
    
def get_rlay_fp_d(ar_d, layName, tmp_path):
    ofp_d = dict()
    for scale, ar_raw in ar_d.items():
        ofp = os.path.join(tmp_path, '%s_%03i.tif'%(layName, scale))
        
        width, height = ar_raw.shape
        
        write_array(ar_raw, ofp, crs=crs,transform=rio.transform.from_bounds(*bbox_base.bounds,width, height),  
                     masked=False, nodata=-9999)
        
        ofp_d[scale]=ofp
    
    return ofp_d
 
    
 
@pytest.fixture(scope='function')    
def ar_d(shape, dsc_l, layName):
    """building some dummy grids for a set of scales"""
    return get_ar_d(shape, dsc_l, layName)
    
def get_ar_d(shape, dsc_l, layName):
    assert dsc_l[0]==1
    d1 = shape[0]
    assert d1%dsc_l[-1]==0, 'bad divisor'
    #===========================================================================
    # build base
    #===========================================================================
    s1 = (10, 10)
    if layName=='wd':        
        samp_ar = np.concatenate( #50% real 50% ry
            (np.round(np.random.random(s1)*10, 2).ravel(),
            np.full(s1, 0).ravel())
            ).ravel()
            
    elif layName=='wse':
        samp_ar = np.concatenate( #50% real 50% ry
            (np.round(np.random.random(s1)*20, 2).ravel(),
            np.full(s1, -9999).ravel())
            ).ravel()
            
    elif layName=='catMosaic':
        samp_ar = np.array(list(cm_int_d.values()))
    
    else:
        raise IOError('not implemented')
    
 
    #===========================================================================
    # random sample from these
    #===========================================================================
    res_d = dict()
    for scale in dsc_l: 
        assert d1%scale==0, 'bad divisor: %i'%scale
        si = tuple((np.array(shape)//scale).astype(int))           
        res_d[scale] = np.random.choice(samp_ar, size=si)
        
        
    return res_d

    

#===============================================================================
# TESTS-------------
#===============================================================================

@pytest.mark.parametrize('finv_fp', [proj_d['finv_fp']])
@pytest.mark.parametrize('shape', [(10,10)], indirect=False)
@pytest.mark.parametrize('bbox', [
                                bbox1, 
                                None
                                  ])
def test_01_assetRsc(wrkr, cMask_pick_fp, finv_fp, bbox): 
    ofp = wrkr.build_assetRsc(cMask_pick_fp, finv_fp, bbox=bbox)
    



@pytest.mark.parametrize('dsc_l', [([1,2,5])])
@pytest.mark.parametrize('layName', [
    #'wd',
    'wse'])
@pytest.mark.parametrize('finv_fp', [proj_d['finv_fp']])
@pytest.mark.parametrize('shape', [(10,10)], indirect=False)
@pytest.mark.parametrize('bbox', [
                                bbox1, 
                                None
                                  ])
def test_02_laySamp(wrkr, lay_pick_fp, finv_fp, bbox, layName): 
    ofp = wrkr.build_layerSamps(lay_pick_fp, finv_fp, bbox=bbox, layName=layName,
                                write=True,
                                )
    
@pytest.mark.dev
@pytest.mark.parametrize('proj_d', [proj_d])
@pytest.mark.parametrize('dsc_l', [([1,2,5])])
@pytest.mark.parametrize('shape', [(10,10)], indirect=False)
def test_runExpo(proj_d, tmp_path, complete_pick_fp):
    run_expo( wrk_dir=tmp_path, case_name='tCn', run_name='tRn', proj_d=proj_d,
              fp_d={'catMasks':complete_pick_fp},
              )
    
 
    
    
    
    
    
    
    
    