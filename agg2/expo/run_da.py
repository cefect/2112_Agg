'''
Created on Sep. 10, 2022

@author: cefect
'''
import os, pathlib
from definitions import proj_lib
from hp.basic import get_dict_str
import pandas as pd
idx = pd.IndexSlice

def SJ_plots_0910(
        fp_lib = {
                'direct':{
                      
                    'arsc':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\direct\20220910\arsc\SJ_r1_direct_0910_arsc.pkl',
                    },
                'filter':{
 
                    'arsc':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r1\SJ\filter\20220910\arsc\SJ_r1_filter_0910_arsc.pkl',
                    }
            }
        ):
    return run_plots(fp_lib)

def run_plots(fp_lib,
                  **kwargs):
    """construct figure from SJ downscale cat results"""
    from agg2.expo.da import ExpoDASession as Session
    from hp.pd import view
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    out_dir = pathlib.Path(os.path.dirname(fp_lib['filter']['arsc'])).parents[3] #C:/LS/10_OUT/2112_Agg/outs/agg2/r5
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, **kwargs) as ses:
        
        #=======================================================================
        # data prep
        #=======================================================================
        #join the simulation results (and clean up indicides
        dxcol_raw = ses.join_arsc_stats(fp_lib, write=False)
        
        
        
        
        
        
if __name__ == "__main__":
    SJ_plots_0910()