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
        studyArea_l = ['obwb', 'noise'], #convenience filtering of proj_lib
        proj_lib = None,
        
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
    #===========================================================================
    # execute
    #===========================================================================
    with RastRun(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                     write=write,exit_summary=exit_summary,prec=prec,
                 bk_lib = {
                     'drlay_lib':dict( severity=severity, downSampling=downSampling, dsampStage=dsampStage, iters=iters),
                     'res_dx':dict(phase_l=phase_l),           
                     },
                 **kwargs) as ses:
        
        if 'depth' in phase_l:
            ses.runDownsample()
        
        if 'diff' in phase_l:
            ses.runDiffs()
        
        ses.retrieve('res_dx')
 
        
        if write_lib:
            ses.write_lib(compression=compression, 
                          id_params=dict(downSampling=downSampling, dsampStage=dsampStage, severity=severity), 
                          debug_max_len=debug_max_len, phase_l=phase_l)

    print('\nfinished %s'%tag)
    
    return 


def dev():
    return run(
        trim=True, compression='none',name='hydR_dev',
        tag='dev',
        iters=2,
        dsampStage='pre',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\dev\dev\20220513\working\drlay_lib_dev_dev_0513.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\dev\dev\20220513\working\noData_cnt_dev_dev_0513.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\dev\dev\20220513\working\rstats_dev_dev_0513.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\dev\dev\20220513\working\wetStats_dev_dev_0513.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\dev\dev\20220513\working\gwArea_dev_dev_0513.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\dev\dev\20220513\working\res_dx_dev_dev_0513.pickle',

            },
        #studyArea_l=['obwb'],
        phase_l=['depth']
        )

 

 
def r7(**kwargs):
    rkwargs = dict(
        iters=3, downSampling='Average',write_lib=False, 
        )    
    return run(name='hr7', **{**rkwargs, **kwargs})

 
    
def r7_post():
    return r7(tag='post', dsampStage='post',  
               compiled_fp_d = {

                }, )

if __name__ == "__main__": 
    
    dev()
    #r7_post()
    #r5_dep()

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))