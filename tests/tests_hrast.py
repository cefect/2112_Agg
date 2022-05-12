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

from agg.hydR.hr_scripts import RastRun

@pytest.fixture(scope='function')
def rastRun(tmp_path,
            #wrk_base_dir=None, 
            base_dir, write,logger, feedback,#see conftest.py (scope=session)

                    ):
    np.random.seed(100)
    with RastRun(out_dir = tmp_path,  wrk_dir=None, 
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        
        assert len(ses.data_d)==0
        yield ses

def test_rstats(rastRun):
    pass