'''
Created on Sep. 26, 2022

@author: cefect
'''
import os, pathlib, itertools, logging, sys
import pandas as pd
import numpy as np
from hp.basic import get_dict_str, today_str, lib_iter
from hp.pd import append_levels, view
 


from agg2.da import CombinedDASession as Session
from agg2.coms import log_dxcol
#===============================================================================
# setup matplotlib----------
#===============================================================================
cm = 1/2.54
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
# globals
#===============================================================================
res_fp_lib = {
    'r11':{
        'haz':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r11\SJ\da\haz\20220930\SJ_r11_direct_0930_dprep.pkl',
        
        },
    
    'r10':
              {
            'haz': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\da\haz\20220926\SJ_r10_direct_0926_dprep.pkl',
            #'exp':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\da\20220926\bstats\SJ_r8_expo_da_0926_bstats.pkl'
            'exp':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\da\20220927\bstats\SJ_r8_expo_da_0927_bstats.pkl'
            }
              }

 

def SJ_da_run(
        run_name='r10',
        **kwargs):    
    
    return run_plots_combine(res_fp_lib[run_name], proj_name='SJ', run_name=run_name, **kwargs)



 
    
def run_plots_combine(fp_lib,pick_fp=None,write=True,**kwargs):
    """da and figures which combine hazard and exposure
    
    
    
    """
 
 
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    out_dir = os.path.join(pathlib.Path(os.path.dirname(fp_lib['exp'])).parents[1],'da', 'expo',today_str)
    print('out_dir:   %s'%out_dir)
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, write=write,logger=logging.getLogger('r'), **kwargs) as ses:
        log = ses.logger 
        
        if pick_fp is None:
            dx1 = ses.build_combined(fp_lib)
        else:
            dx1 = pd.read_pickle(pick_fp)

        #=======================================================================
        # plots-------
        #=======================================================================
 
        #combined_single(ses, dx1)
        
        ses.plot_4x4_subfigs(dx1)
        
        

if __name__ == "__main__":
    SJ_da_run()
    #SJ_combine_plots_0919()
    
    print('finished')
        