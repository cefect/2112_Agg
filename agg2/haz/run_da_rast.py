'''
Created on Sep. 28, 2022

@author: cefect

data analysis on raster data
'''
import os, pathlib, math, pprint
from definitions import proj_lib

import pandas as pd
idx = pd.IndexSlice
from hp.pd import view
from hp.basic import get_dict_str, today_str

from agg2.haz.run import res_fp_lib
from agg2.haz.da import Session_haz_da_rast as Session
#===============================================================================
# setup matplotlib----------
#===============================================================================
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
 
#font
matplotlib.rc('font', **{
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8})
 
 
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(10,10),
    'legend.title_fontsize':'large'
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)

def SJ_da_rast_run(
        run_name='r10',
        **kwargs):    
    
    return run_rast_plots(res_fp_lib[run_name], proj_name='SJ', run_name=run_name, **kwargs)


def run_rast_plots(fp_lib,
                  pick_fp=None,
                  write=True,
                  **kwargs):
    """construct figure from SJ downscale cat results"""

    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    """combines filter and direct"""
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(fp_lib['filter'])).parents[3],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
        'da', 'haz', today_str)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, **kwargs) as ses:
        """for haz, working with aggregated zonal stats.
            these are computed on:
                aggregated (s2) data with UpsampleSession.run_stats()
                raw/fine (s1) data with UpsampleSession.run_stats_fine()
                local diff (s2-s1) with UpsampleSession.run_diff_stats()
            
 
        """
        idxn = ses.idxn
        log = ses.logger
        
        
if __name__ == "__main__":

  
    SJ_da_rast_run(run_name='r10')
    
    print('finished')