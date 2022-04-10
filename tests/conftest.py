'''
Created on Feb. 21, 2022

@author: cefect
'''
import os, shutil
import pytest

    
#===============================================================================
# fixture-----
#===============================================================================
@pytest.fixture(scope='session')
def write():
    write=True
    if write:
        print('WARNING!!! runnig in write mode')
    return write

@pytest.fixture(scope='session')
def logger():
    out_dir = r'C:\LS\10_OUT\2112_Agg\outs\tests'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    os.chdir(out_dir) #set this to the working directory
    print('working directory set to \"%s\''%os.getcwd())

    from hp.logr import BuildLogr
    lwrkr = BuildLogr()
    return lwrkr.logger

@pytest.fixture(scope='session')
def feedback(logger):
    from hp.Q import MyFeedBackQ
    return MyFeedBackQ(logger=logger)
    
 
    



@pytest.fixture(scope='session')
def base_dir():
    base_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\compiled'
    assert os.path.exists(base_dir)
    return base_dir



@pytest.fixture
def true_dir(write, tmp_path, base_dir):
    true_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    if write:
        if os.path.exists(true_dir): 
            shutil.rmtree(true_dir)
            os.makedirs(true_dir) #add back an empty folder
            
    return true_dir
    
    
#===============================================================================
# helpers-------
#===============================================================================
def build_compileds(dkey_d, base_dir): 
    d = dict()
    for dkey, folder in dkey_d.items():
        input_fp = search_fp(os.path.join(base_dir, folder), '.pickle', dkey) #find the data file.
        d[dkey] = input_fp
    return d

def retrieve_finv_d(finv_agg_fn, session, base_dir):
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
    assert os.path.exists(dirpath), 'searchpath does not exist: %s'%dirpath
    fns = [e for e in os.listdir(dirpath) if e.endswith(ext)]
    
    result= None
    for fn in fns:
        if fn.startswith(pattern):
            result = os.path.join(dirpath, fn)
            break
        
    if result is None:
        raise IOError('failed to find a match for \'%s\' in %s'%(pattern, dirpath))
    
    assert os.path.exists(result), result
        
        
    return result