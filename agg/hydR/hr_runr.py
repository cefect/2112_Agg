'''
Created on May 8, 2022

@author: cefect
'''
import os, datetime, math, pickle, copy, sys
import numpy as np
np.random.seed(100)

start = datetime.datetime.now()
print('start at %s' % start)
from agg.hydR.hr_scripts import RastRun


def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='tag',
        name='hrast0',
        overwrite=True,
        trim=False,

        #=======================================================================
        # write control
        #=======================================================================
        write=True,
        exit_summary=True,
        write_lib=True, #enter the results into the library
 
        compression='med',
        #=======================================================================
        # #data
        #=======================================================================
        studyArea_l = None, #convenience filtering of proj_lib
        proj_lib = None,
        
        #optional loading data from the catalog
        catalog_fp=None,
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

        #=======================================================================
        # #parameters
        #=======================================================================
        iters=3, #resolution iterations
        #raster downSampling and selection  (StudyArea.get_raster())
        dsampStage='pre', downSampling='Average', severity = 'hi', 
        #resolution=5, this is what we iterate on

        #=======================================================================
        # debug
        #=======================================================================
        debug_max_len=None,phase_l=['depth'],

        **kwargs):
    print('START run w/ %s.%s and '%(name, tag))
 
    #===========================================================================
    # study area filtering
    #===========================================================================
    if proj_lib is None:
        from definitions import proj_lib
    
    if not studyArea_l is None:
        print('filtering studyarea to %i: %s'%(len(studyArea_l), studyArea_l))
        miss_l = set(studyArea_l).difference(proj_lib.keys())
        assert len(miss_l)==0, 'passed %i studyAreas not in proj_lib: %s'%(len(miss_l), miss_l)
        proj_lib = {k:v for k,v in proj_lib.items() if k in studyArea_l}
        
    id_params=dict(downSampling=downSampling, dsampStage=dsampStage, severity=severity)
    #===========================================================================
    # execute
    #===========================================================================
    with RastRun(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                     write=write,exit_summary=exit_summary,prec=prec, 
                 bk_lib = {
                     'drlay_lib':dict( severity=severity, downSampling=downSampling, dsampStage=dsampStage, iters=iters),
                     'res_dx':dict(phase_l=phase_l), 
                     'layxport':dict(compression=compression, debug_max_len=debug_max_len, phase_l=phase_l, id_params=id_params), 
          
                     },
                 **kwargs) as ses:
        
        #
        if not catalog_fp is None:
            ses.compileFromCat(catalog_fp=catalog_fp,id_params=id_params)
            
 
        if 'depth' in phase_l:
            ses.runDownsample()
        
        if 'diff' in phase_l:
            ses.runDiffs()
        
        ses.retrieve('res_dx')
 
        
        if write_lib:
            ses.write_lib(id_params=id_params)

    print('\nfinished %s'%tag)
    
    return 


def dev():
    return run(
        trim=True, compression='none',name='hydR_dev',
        tag='dev',
        iters=2,
        dsampStage='postFN',
        downSampling='Nearest neighbour',
        compiled_fp_d={
            'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR_dev\dev\20220515\working\drlay_lib_hydR_dev_dev_0515.pickle',

            },
        #studyArea_l=['obwb'],
        phase_l=['depth']
        )

 

 
def r01(**kwargs):
    rkwargs = dict(
        iters=8, downSampling='Average',write_lib=True, 
        #catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydR01\hydR01_run_index.csv',
        )    
    return run(name='hydR01', **{**rkwargs, **kwargs})

def postFN():
    return r01(
        dsampStage='postFN',tag='postFN',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\drlay_lib_hydR01_postFN_0514.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\noData_cnt_hydR01_postFN_0514.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\rstats_hydR01_postFN_0514.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\wetStats_hydR01_postFN_0514.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\gwArea_hydR01_postFN_0514.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\res_dx_hydR01_postFN_0514.pickle',
        'layxport':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\layxport_hydR01_postFN_0514.pickle',
            }
        )

def post():
    return r01(
        dsampStage='post',tag='post',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\drlay_lib_hydR01_post_0514.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\noData_cnt_hydR01_post_0514.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\rstats_hydR01_post_0514.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\wetStats_hydR01_post_0514.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\gwArea_hydR01_post_0514.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\res_dx_hydR01_post_0514.pickle',
        'layxport':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\layxport_hydR01_post_0514.pickle',
            }
        )
    
def pre():
    return r01(
        dsampStage='pre',tag='pre',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\pre\20220514\working\drlay_lib_hydR01_pre_0514.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\pre\20220514\working\noData_cnt_hydR01_pre_0514.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\pre\20220514\working\rstats_hydR01_pre_0514.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\pre\20220514\working\wetStats_hydR01_pre_0514.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\pre\20220514\working\gwArea_hydR01_pre_0514.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\pre\20220514\working\res_dx_hydR01_pre_0514.pickle',
        'layxport':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\pre\20220514\working\layxport_hydR01_pre_0514.pickle',
            }
        )
    
def preGW():
    return r01(
        dsampStage='preGW',tag='preGW',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\drlay_lib_hydR01_preGW_0514.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\noData_cnt_hydR01_preGW_0514.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\rstats_hydR01_preGW_0514.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\wetStats_hydR01_preGW_0514.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\gwArea_hydR01_preGW_0514.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\res_dx_hydR01_preGW_0514.pickle',
        'layxport':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\layxport_hydR01_preGW_0514.pickle',
            }
        )
    
 
if __name__ == "__main__": 
    
    dev()
 
    #post()
    #postFN()
    #pre()
    #preGW()
    #r5_dep()

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))