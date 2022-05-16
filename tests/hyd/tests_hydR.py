'''
Created on May 12, 2022

@author: cefect

testing hrast specific callers
    see hyd.tests_expo for related tests
'''
import os  
import pytest
print('pytest.__version__:' + pytest.__version__)
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
idx = pd.IndexSlice

from hp.gdal import array_to_rlay
from agg.hydR.hydR_scripts import RastRun

prec=3

class Session(RastRun):
    
    def ar2lib(self, #get a container of a raster layer in RastRun style
               ar,
               resolution=10.0,
               studyArea='testSA'):
        
        log = self.logger.getChild('ar2lib')
        rlay_fp = array_to_rlay(ar, resolution=resolution)
        rlay =  self.get_layer(rlay_fp, mstore=self.mstore)
        log.info('built %s to \n    %s'%(ar.shape, rlay_fp))
        
        return {resolution:{studyArea:rlay}}
        

@pytest.fixture
def ses(tmp_path,write,logger, feedback, #session for depth runs
        array,
        resolution):
 
    np.random.seed(100)
    with Session(out_dir = tmp_path,  wrk_dir=None, prec=prec,
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        
        assert len(ses.data_d)==0
        ses.data_d['drlay_lib'] = ses.ar2lib(array, resolution=resolution)
 
        yield ses
        


        
@pytest.fixture
def resolution(request):
    return request.param

@pytest.fixture
def array(request):
    return request.param
 

#===============================================================================
# DEPTH RASERTS------
#===============================================================================

@pytest.mark.parametrize('resolution', [10.0])
@pytest.mark.parametrize('array',[
     np.random.random((5,5)),
     np.array([(1,np.nan),(1,2)]), 
     ], indirect=True)  
def test_rstats(ses, array):
    ar = array.copy()
    #===========================================================================
    # raster calc
    #===========================================================================
    dkey='rstats'
    
    rserx = ses.build_stats(dkey=dkey).iloc[0,:]
    
 
    #===========================================================================
    # check
    #===========================================================================
    #methods
    df = pd.DataFrame(ar)
 
    for rsColn, npMeth in {
        'MAX':'max',  'MIN':'min', 'SUM':'sum'
        }.items():
        """numpy methods dont have nan support
        assert round(rserx[rsColn], prec) ==  round(getattr(ar, npMethod)(), prec), rsColn"""
        ser = getattr(df, npMeth)(axis=1)
        npVal = round(getattr(ser, npMeth)(), prec)
        assert round(rserx[rsColn], prec) ==  npVal, rsColn
        
    #average
    """sequence matters here... cant use builtin"""
    npVal=df.sum().sum()/df.notna().sum().sum()
    assert round(rserx['MEAN'], prec) ==  round(npVal, prec), rsColn
 
    #nodata
    assert np.isnan(ar).sum()==rserx['noData_cnt']


@pytest.mark.parametrize('resolution', [10.0])
@pytest.mark.parametrize('array',[
     np.random.random((5,5))-0.5,
     #np.array([(1,np.nan),(1,2)]),  #depth raster... no nulls
     ], indirect=True)  
def test_wetStats(ses, array, resolution):
    dkey='wetStats'
    rserx=ses.retrieve(dkey).iloc[0,:]
    
    wetArray = array[array>0.0]
    
    #wetcount
    wet_cnt = (array>0.0).sum()
    assert wet_cnt==rserx['wetCnt']
    
    #wetArea
    assert rserx['wetArea']==wet_cnt*(resolution**2)
    
    
    #volumes
 
    npVol = wetArray.sum()*(resolution**2)
    assert round(rserx['wetVolume'], prec)==round(npVol, prec)
    
    #mean
    npMean = wetArray.mean()
    assert round(rserx['wetMean'], prec)==round(npMean, prec)
    
    
 
@pytest.mark.parametrize('resolution', [10.0])
@pytest.mark.parametrize('array',[
     np.random.random((5,5))-0.5,
     #np.array([(1,np.nan),(1,2)]),  #depth raster... no nulls
     ], indirect=True)  
def test_gwArea(ses, array, resolution):
    dkey='gwArea'
    rserx=ses.retrieve(dkey).iloc[0,:]
    
    #wetcount
    wet_cnt = (array<0.0).sum()
 
    
    #wetArea
    assert rserx['gwArea']==wet_cnt*(resolution**2)

#===============================================================================
# DIFFERENCE RASTERS---------
#===============================================================================
@pytest.fixture
def ses_diff(tmp_path,write,logger, feedback, #session for difference rastser runs
        array,
        resolution):
 
    np.random.seed(100)
    with Session(out_dir = tmp_path,  wrk_dir=None, prec=prec,
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        
        assert len(ses.data_d)==0
        ses.data_d['difrlay_lib'] = ses.ar2lib(array, resolution=resolution)
 
        yield ses
        
        
@pytest.mark.dev
@pytest.mark.parametrize('resolution', [10.0])
@pytest.mark.parametrize('array',[
     np.random.random((5,5))-0.5,
     np.random.random((1,2))*10,
     np.random.random((5,5)),
     #np.array([(1,np.nan),(1,2)]),  #depth raster... no nulls
     ], indirect=True)  
def test_rmseD(ses_diff, array, resolution):
    """
    RastRun.build_rmseD()
    """
    dkey='rmseD'
    rserx=ses_diff.retrieve(dkey).iloc[0,:]
 
    rmse = np.sqrt(np.mean(np.square(array)))
    
 
 
    
    #wetArea
    assert round(rserx[0], prec)==round(rmse, prec)
 
     
 