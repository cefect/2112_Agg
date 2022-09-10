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
        arsc_dx = ses.join_arsc_stats(fp_lib, write=False)
        
        #=======================================================================
        # stackced areas ratios
        #=======================================================================
        
        #=======================================================================
        # #compute fraction
        #=======================================================================
        coln = 'frac'
        dxcol1 = arsc_dx.droplevel('metric', axis=1)
         
        #divide by the total        
        cnt_dx = dxcol1.divide(cnt_dx.loc[:, idx[:, 'all', :]].droplevel([1,2], axis=1), axis=0, level=0).droplevel(-1, axis=1)
         
        #add the new label
        cnt_dx = pd.concat([cnt_dx], keys=[coln], names=['metric'], axis=1).reorder_levels(dxcol4.columns.names, axis=1)
         
        dxcol4 = dxcol4.join(cnt_dx).sort_index(axis=1)
          
        #slice to just these
        dfi = dxcol4.loc[:, idx['direct', :, 'frac']].droplevel([0,2], axis=1).drop('all', axis=1)
         
        #plot
        ses.plot_dsc_ratios(dfi.dropna())
        
        
        
        
if __name__ == "__main__":
    SJ_plots_0910()