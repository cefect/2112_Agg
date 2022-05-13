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
        studyArea_l = ['obwb'], #convenience filtering of proj_lib
        proj_lib = None,
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

        #=======================================================================
        # #parameters
        #=======================================================================
        #raster downSampling and selection  (StudyArea.get_raster())
        iters=3, #resolution iterations
        dsampStage='pre', downSampling='Average', severity = 'hi', 
        
        
        #aggregation
        aggType = 'convexHulls', aggIters = 3,
        
        #sampling (geo). see Model.build_sampGeo()
        sgType = 'poly', 
        
        #sampling (method). see Model.build_rsamps()
        samp_method = 'zonal', zonal_stat='Mean',  # stats to use for zonal. 2=mean
        
        #outputting
        compression='med',

        #=======================================================================
        # debug
        #=======================================================================
        phase_l=['depth', 'expo'],

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
        
    
    #indexing parameters for catalog
    id_params={
                **dict(downSampling=downSampling, dsampStage=dsampStage, severity=severity),
                **dict(aggType=aggType, samp_method=samp_method)}
    #===========================================================================
    # execute
    #===========================================================================
    with ExpoRun(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                     write=write,exit_summary=exit_summary,prec=prec,phase_l=phase_l,
                 bk_lib = {
                     'drlay_lib':dict( severity=severity, downSampling=downSampling, dsampStage=dsampStage, iters=iters),
                     'finv_agg_lib':dict(aggType=aggType, iters=aggIters),
                     'finv_sg_lib':dict(sgType=sgType),
                     'rsamps':dict(samp_method=samp_method, zonal_stat=zonal_stat),
                     'res_dx':dict(),
                     'layxport':dict(compression=compression, id_params=id_params)
                     },
                 **kwargs) as ses:
        
        #=======================================================================
        # call each phase
        #=======================================================================
        if 'depth' in phase_l:
            ses.runDownsample()
        
        if 'diff' in phase_l:
            ses.runDiffs()
            
        if 'expo' in phase_l:
            ses.runExpo()
 
        #=======================================================================
        # write results to library
        #=======================================================================
        if write_lib:
            ses.write_lib(id_params=id_params)

    print('\nfinished %s'%tag)
    
    return 


def dev():
    return run(
        trim=True, name='hydE_dev',
        tag='dev',
 
 
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\drlay_lib_hydE_dev_dev_0513.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\noData_cnt_hydE_dev_dev_0513.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\rstats_hydE_dev_dev_0513.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\wetStats_hydE_dev_dev_0513.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\gwArea_hydE_dev_dev_0513.pickle',
        'finv_agg_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\finv_agg_lib_hydE_dev_dev_0513.pickle',
        'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\finv_agg_mindex_hydE_dev_dev_0513.pickle',
        'finv_sg_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\finv_sg_lib_hydE_dev_dev_0513.pickle',
        'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\rsamps_hydE_dev_dev_0513.pickle',
        #'rsampStats':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\rsampStats_hydE_dev_dev_0513.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\res_dx_hydE_dev_dev_0513.pickle',
        'layxport':r'C:\LS\10_OUT\2112_Agg\outs\hydE_dev\dev\20220513\working\layxport_hydE_dev_dev_0513.pickle',


             },
        #studyArea_l=['obwb'],
        #phase_l=['depth']
        )

 

if __name__ == "__main__": 
    
    dev()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))