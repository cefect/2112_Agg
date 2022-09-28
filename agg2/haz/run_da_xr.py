'''
Created on Sep. 28, 2022

@author: cefect

data analysis on raster data
'''
import os, pathlib, math, pprint, logging, sys
from definitions import proj_lib
from rasterio.crs import CRS

import pandas as pd
idx = pd.IndexSlice
from hp.pd import view
from hp.basic import get_dict_str, today_str

from agg2.haz.run_stats import xr_lib
from agg2.haz.da_xr import Session_haz_da_rast as Session
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


#===============================================================================
# setup logger
#===============================================================================
logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )

#===============================================================================
# funcs--------
#===============================================================================

def SJ_da_rast_run(
        run_name='r10',
        crs=None,
        case_name='SJ',
        **kwargs): 
 
    if crs is None:
        crs = CRS.from_epsg(proj_lib[case_name]['EPSG'])   
    
    return run_rast_plots(xr_lib[run_name], case_name=case_name, run_name=run_name,crs=crs,
                           **kwargs)


def run_rast_plots(xr_dir_lib,
                  pick_fp=None,
                  write=True,
                  **kwargs):
    """construct figure from SJ downscale cat results"""

    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    """combines filter and direct"""
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(xr_dir_lib['direct'])).parents[1],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
        'da', 'rast', today_str)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir,logger=logging.getLogger('r'), **kwargs) as ses:
        """for haz, working with aggregated zonal stats.
            these are computed on:
                aggregated (s2) data with UpsampleSession.run_stats()
                raw/fine (s1) data with UpsampleSession.run_stats_fine()
                local diff (s2-s1) with UpsampleSession.run_diff_stats()
            
 
        """
        idxn = ses.idxn
        log = ses.logger
        
        #=======================================================================
        # loop on each method
        #=======================================================================
        #build a datasource from the netcdf files
        for method, xr_dir in xr_dir_lib.items():
            #===================================================================
            # load data
            #===================================================================
            ds = ses.get_ds_merge(xr_dir)
            
            #===================================================================
            # plot gaussian of wd
            #===================================================================
            #data prep
            dar = ds['wd']
            dar1 = dar.where(dar[0]!=0)
            """
            dar1.plot(col='scale')
            """
                        
            ses.plot_gaussian_set(dar1)
        
        
if __name__ == "__main__":

  
    SJ_da_rast_run(run_name='dev')
    
    print('finished')