'''
Created on Mar. 18, 2023

@author: cefect

tests for the hydro-downscale compare
'''

import pytest, copy, os, random, re
xfail = pytest.mark.xfail
import numpy as np
import pandas as pd
import shapely.geometry as sgeo
from rasterio.enums import Resampling

from hcomp.scripts import HydCompareSession
from hcomp.run_hcomp import run_compare


from tests.hcomp.conftest import get_aoi_fp
from hp.tests.tools.rasters import get_rlay_fp
#===============================================================================
# test data------
#===============================================================================
 
from tests.hcomp.data.toy import (
    dem1_ar, wse1_ar, wsh1_ar, wse2_ar, crs_default,  bbox_default,
    )

gfp = lambda ar, name:get_rlay_fp(ar, name, crs=crs_default, bbox=bbox_default)

dem1_fp = gfp(dem1_ar, 'dem1') 
wse1_fp = gfp(wse1_ar, 'wse1')
wsh1_fp = gfp(wsh1_ar, 'wsh1')
wse2_fp = gfp(wse2_ar, 'wse2')
 
aoi_fp = get_aoi_fp(sgeo.box(0, 0, 6, 9), crs=crs_default)



#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(init_kwargs):
    
    """Mock session for tests""" 
    with HydCompareSession(**init_kwargs) as ses: 
        yield ses

#===============================================================================
# tests-------
#===============================================================================
pars_filepaths = 'dem_fp, wsh_fp, wse_fp', [
    (dem1_fp,wsh1_fp, wse1_fp),
    ]


@pytest.mark.parametrize(*pars_filepaths)
@pytest.mark.parametrize('aggscale', [3])
@pytest.mark.parametrize('resampling', [Resampling.average])
def test_get_avgWSH(dem_fp, wsh_fp, wse_fp, aggscale, resampling, wrkr): 
    wrkr.get_avgWSH(dem_fp, wsh_fp,  aggscale=aggscale, resampling=resampling)


@pytest.mark.parametrize(*pars_filepaths)
@pytest.mark.parametrize('aggscale', [3])
@pytest.mark.parametrize('resampling', [Resampling.average])
def test_get_avgWSE(dem_fp, wsh_fp, wse_fp, aggscale, resampling, wrkr): 
    wrkr.get_avgWSE(dem_fp, wse_fp,  aggscale=aggscale, resampling=resampling)


@pytest.mark.parametrize(*pars_filepaths)
@pytest.mark.parametrize('aggscale', [3])
@pytest.mark.parametrize('method, mkwargs', [
    ('avgWSH', dict(resampling=Resampling.average)),
    ('avgWSE', dict(resampling=Resampling.average)),
    ])
def test_agg_byType(dem_fp, wsh_fp, wse_fp, method, mkwargs, aggscale, wrkr): 
    wrkr.get_agg_byType(method, dem_fp=dem_fp, wse_fp=wse_fp, wsh_fp=wsh_fp, 
                        aggscale=aggscale, method_kwargs=mkwargs)

@pytest.mark.dev
@pytest.mark.parametrize('dem1_fp, wse1_fp, wse2_fp, aoi_fp', [
    (dem1_fp,wse1_fp, wse2_fp, aoi_fp),
    ])
def test_run_compare(dem1_fp,wse1_fp, wse2_fp, aoi_fp, init_kwargs):
    run_compare(dem1_fp=dem1_fp, wse1_fp=wse1_fp, wse2_fp=wse2_fp,
                aoi_fp=aoi_fp,
                init_kwargs=init_kwargs
                )
                
