'''
unit tests for downsample v2
'''
 
from hp.np import dropna
from hp.rio import RioWrkr, write_array, load_array
from numpy import array, dtype
from tests2.conftest import validate_dict, src_dir, get_abs
import numpy as np
import pandas as pd
import pytest, copy, os, random
import rasterio as rio
from agg2.haz.coms import get_rand_ar, get_wse_filtered, assert_dx_names
from agg2.haz.scripts import UpsampleSession as Session
from agg2.haz.run import run_haz_agg2
 
xfail = pytest.mark.xfail
 

#===============================================================================
# helpers and globals------
#===============================================================================
crsid = 2953
prec=5
crs=rio.crs.CRS.from_epsg(crsid)
#for test data
output_kwargs = dict(crs=crs,transform=rio.transform.from_origin(1,100,1,1)) 

test_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data'

def assert_stat_check(fp, msg=''):
    dx = pd.read_pickle(fp)
    assert_dx_names(dx, msg=msg+' %s'%os.path.basename(fp))
    
#===============================================================================
# FIXTURES-----
#===============================================================================
@pytest.fixture(scope='function')
def dem_fp(dem_ar, tmp_path):
    if dem_ar is None:
        return None
    return write_array(dem_ar, os.path.join(tmp_path, 'dem.tif'), **output_kwargs)
 

@pytest.fixture(scope='function')
def wse_fp(wse_ar, tmp_path): 
    return write_array(wse_ar, os.path.join(tmp_path, 'wse.tif'), **output_kwargs)

@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name, 
                    ):
    
    """Mock session for tests"""
 
    np.random.seed(100)
    random.seed(100)
 
    
    with Session(
                 crs=crs, nodata=-9999,
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                 prec=prec,
 
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

@pytest.mark.parametrize('dem_ar, wse_ar', [
    get_rand_ar((16,16))
    ]) 
@pytest.mark.parametrize('dsc_l', [([1,2,4])])
@pytest.mark.parametrize('method', [
    'direct', 
    'filter'])
def test_00_runDsmp(wrkr, dsc_l,method,
                    dem_fp,dem_ar,wse_fp, wse_ar,
                    ):
    
    wrkr.run_agg(dem_fp, wse_fp,  dsc_l=dsc_l,
                  #write=True,
                  #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data', method).
                  )


@pytest.mark.parametrize('reso_iters', [3, 10])
def test_00_dscList(wrkr, reso_iters):
    
    #build with function
    res_l = wrkr.get_dscList(reso_iters=reso_iters)
    
    #===========================================================================
    # #validate
    #===========================================================================
    assert isinstance(res_l, list)
    assert len(res_l)==reso_iters
    assert res_l[0]==1

 

@pytest.mark.parametrize('dem_ar, wse_ar', [
    get_rand_ar((8,8))
    ])
@pytest.mark.parametrize('method', [
    'direct', 
    #'filter',
    ])
@pytest.mark.parametrize('dsc_l', [([1,2])])
def test_01_dset(dem_fp,dem_ar,wse_fp, wse_ar,   wrkr, dsc_l, method):
    wrkr.build_dset(dem_fp, wse_fp, dsc_l=dsc_l, method=method)


@pytest.mark.parametrize('dem_ar, wse_ar', [
    get_rand_ar((8,8))
    ])
@pytest.mark.parametrize('method', [
    'direct', 
    'filter',
    ])
@pytest.mark.parametrize('dsc_l', [([1,2])])
def test_01_runAgg(dem_fp,dem_ar,wse_fp, wse_ar,   wrkr, dsc_l, method):
    """wrapper for build_dset"""
    wrkr.run_agg(dem_fp, wse_fp, method=method, dsc_l=dsc_l, write=True,
                 #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data')
                 )
agg_fp = os.path.join(src_dir, r'tests2\haz\data\agg_filter\dsTest_test01_0908_agg_filter.pkl')



@pytest.mark.parametrize('pick_fp', [
    agg_fp,
     #os.path.join(src_dir, r'tests2\haz\data\filter\dsTest_test00_0828_haz_dsmp.pkl'),
     ])
def test_02_dsc(wrkr, pick_fp):
    res_fp = wrkr.run_catMasks(pick_fp, write=True,
                               #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data')
                               )
    

cmasks_fp = os.path.join(src_dir, r'tests2\haz\data\cMasks\dsTest_test02_0908_cMasks.pkl')

@pytest.mark.dev
@pytest.mark.parametrize('pick_fp', [cmasks_fp]) 
def test_03_stats(wrkr, pick_fp):
    res_fp = wrkr.run_stats(pick_fp, write=True)
    assert_stat_check(res_fp)



@pytest.mark.parametrize('pick_fp', [cmasks_fp]) 
def test_04_statsFine(wrkr, pick_fp):
    res_fp = wrkr.run_stats_fine(pick_fp, write=True)
    assert_stat_check(res_fp)

 

@pytest.mark.parametrize('pick_fp', [cmasks_fp]) 
def test_05_errs(wrkr, pick_fp):
    res_fp = wrkr.run_errs(pick_fp, write=True,
                           #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data')
                           )

    
err_fp = os.path.join(src_dir, 'tests2\haz\data\errs\dsTest_test05_0909_errs.pkl')
    

@pytest.mark.parametrize('pick_fp', [err_fp]) 
def test_06_errStats(wrkr, pick_fp):
    res_fp = wrkr.run_errStats(pick_fp, write=True,
                           #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data')
                           )
    assert_stat_check(res_fp)
    
    
#===============================================================================
# INTEGRATIOn tests ------------
#===============================================================================
#saint Jon sub-set test data
SJ_test_dir= r'C:\LS\10_OUT\2112_Agg\ins\hyd\SaintJohn\test'

proj_d = {
    'EPSG':2953,
    'finv_fp':'',
    'wse_fp_d':{'hi':os.path.join(SJ_test_dir,  'GeoNB_LSJ_aoiT01_0829.tif')},
    'dem_fp_d':{1:os.path.join(SJ_test_dir,'NBDNR2015_r01_aoiT01_0829.tif')},
    }
 


 
    


@pytest.mark.parametrize('dsc_l', [([1,2,4])])
@pytest.mark.parametrize('method', [
    'direct', 
    'filter',
    ])
@pytest.mark.parametrize('proj_d', [proj_d])
def test_runHaz(method, proj_d, dsc_l, tmp_path):
    """use the function runner"""
    fp_d, stat_d = run_haz_agg2(proj_d=proj_d, method=method, dsc_l=dsc_l, case_name='tCn', run_name='tRn',
                 wrk_dir=tmp_path)
    
    for k, fp in stat_d.items():
        assert_stat_check(fp)
    
    
    
    
 
 
