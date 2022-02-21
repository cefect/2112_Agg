'''
Created on Feb. 21, 2022

@author: cefect
'''
import os
import pytest

    
#===============================================================================
# fixture-----
#===============================================================================
@pytest.fixture(scope='session')
def base_dir():
    base_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\compiled'
    assert os.path.exists(base_dir)
    return base_dir

@pytest.fixture(scope='session')
def write():
    write=False
    if write:
        print('WARNING!!! runnig in write mode')
    return write


