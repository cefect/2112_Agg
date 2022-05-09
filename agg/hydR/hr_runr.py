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
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

 
        #=======================================================================
        # #parameters
        #=======================================================================
        iters=3, #resolution iterations
        #raster downSampling and selection  (StudyArea.get_raster())
        dsampStage='wse', downSampling='Average', severity = 'hi', 
        #resolution=5, this is what we iterate on
        
 
        
        #=======================================================================
        # debug
        #=======================================================================
        debug_max_len=None,
 
        
 
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
 
                                          
                     },
                 **kwargs) as ses:
        
        ses.runDownsample()
        
        ses.runDiffs()
        
        if write_lib:
            ses.write_lib(compression=compression, id_params=dict(downSampling=downSampling, dsampStage=dsampStage, severity=severity), debug_max_len=debug_max_len)
        
 
        
 
        
    print('\nfinished %s'%tag)
    
    return 


def dev():
    return run(
        trim=True, compression='none',name='dev',
        tag='dev',
        iters=2,
        compiled_fp_d={
 
            },
        studyArea_l=['obwb'],
        )

 

def r2():
    return run(tag='wse', name='hrast1',iters=10,
               dsampStage='wse', 
               downSampling='Average',
               compiled_fp_d = {
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\drlay_lib_hrast1_wse_0427.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\rstats_basic_hrast1_wse_0427.pickle',
        'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\wetAreas_hrast1_wse_0427.pickle',
 
        
        
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\difrlay_lib_hrast1_wse_0505.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\rstatsD_hrast1_wse_0505.pickle',
        
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\res_dx_hrast1_wse_0505.pickle',
        },
               write_lib=True,
               )
               
def r3_wse():
    return run(tag='wse', name='hrast2',iters=10,
               dsampStage='wse', 
               downSampling='Average',
               compiled_fp_d = {
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\drlay_lib_hrast1_wse_0427.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\rstats_basic_hrast1_wse_0427.pickle',
        'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\wetAreas_hrast1_wse_0427.pickle',
 
        
        
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\difrlay_lib_hrast1_wse_0505.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\rstatsD_hrast1_wse_0505.pickle',
        
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\res_dx_hrast1_wse_0505.pickle',
        },
               write_lib=True,
               #debug_max_len=2,
               )
    
def r3_depth():
    return run(tag='depth', name='hrast2',iters=10,
               dsampStage='depth', 
               downSampling='Average',
               compiled_fp_d = {
                   'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\drlay_lib_hrast2_depth_0505.pickle',
                   'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\difrlay_lib_hrast2_depth_0505.pickle',
                   'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\rstats_hrast2_depth_0505.pickle',
                    'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\wetAreas_hrast2_depth_0505.pickle',
                    'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\rstatsD_hrast2_depth_0505.pickle',
                    'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\res_dx_hrast2_depth_0505.pickle',
                                },
 
               )
    
def r4_wse():
    return run(tag='wse', name='hrast4',iters=7,
               dsampStage='wse', 
               downSampling='Average',
               compiled_fp_d = {
 
                },
               write_lib=False,
               #debug_max_len=2,
               )
if __name__ == "__main__": 
    
    dev()
    #r4_wse()

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))