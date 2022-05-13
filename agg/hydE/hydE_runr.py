'''
Created on May 12, 2022

@author: cefect
'''

'''
Created on May 8, 2022

@author: cefect
'''
import os, datetime, math, pickle, copy, sys
import numpy as np
np.random.seed(100)

start = datetime.datetime.now()
print('start at %s' % start)
from agg.hydE.hydE_scripts import ExpoRun


def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='tag',
        name='hydE',
        overwrite=True,
        trim=False,

        #=======================================================================
        # write control
        #=======================================================================
        write=True,
        exit_summary=True,
        write_lib=True, #enter the results into the library
 
 
        #=======================================================================
        # #data
        #=======================================================================
        studyArea_l = ['obwb', 'LMFRA'], #convenience filtering of proj_lib
        proj_lib = None,
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

        #=======================================================================
        # #parameters
        #=======================================================================
        #aggregation
        aggType = 'convexHulls', aggIters = 3,
        
        #sampling (geo). see Model.build_sampGeo()
        sgType = 'poly', 
        
        #sampling (method). see Model.build_rsamps()
        samp_method = 'zonal', zonal_stat='Mean',  # stats to use for zonal. 2=mean

        #=======================================================================
        # debug
        #=======================================================================
        phase_l=['depth'],

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
    with ExpoRun(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                     write=write,exit_summary=exit_summary,prec=prec,
                 bk_lib = {
                     'finv_agg_d':dict(aggType=aggType, iters=aggIters),
                     'finv_sg_d':dict(sgType=sgType),
                     'rsamps':dict(samp_method=samp_method, zonal_stat=zonal_stat),
                     },
                 **kwargs) as ses:
        
        ses.runExpo()
 
 
        
        if write_lib:
            pass

    print('\nfinished %s'%tag)
    
    return 


def dev():
    return run(
        trim=True, name='hrdev',
        tag='dev',
 
 
        compiled_fp_d={
        'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hrdev\dev\20220513\working\finv_agg_d_hrdev_dev_0513.pickle',
        'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hrdev\dev\20220513\working\finv_agg_mindex_hrdev_dev_0513.pickle',
            },
        #studyArea_l=['obwb'],
        phase_l=['depth']
        )

 

if __name__ == "__main__": 
    
    dev()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))