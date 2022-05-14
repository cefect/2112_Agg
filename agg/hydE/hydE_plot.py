'''
Created on Apr. 26, 2022

@author: cefect

visualizetion on raster calcs
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
start = datetime.datetime.now()
print('start at %s' % start)

import qgis.core
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal
idx = pd.IndexSlice
#===============================================================================
# setup matplotlib
#===============================================================================
 
import matplotlib
matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt

#set teh styles
plt.style.use('default')

#font
matplotlib_font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 14 
matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)

from hydR.hydR_plot import RasterPlotr
from hp.plot import Plotr
from hp.gdal import rlay_to_array, getRasterMetadata
from hp.basic import set_info, get_dict_str
from hp.pd import get_bx_multiVal
#from hp.animation import capture_images

class ExpoPlotr(RasterPlotr): #analysis of model results

    
        #colormap per data type
    colorMap_d = {
 
 
        }
    
    def __init__(self,
 
                 name='hydE_plot',
 
                 **kwargs):
        
 
        
        super().__init__(name=name, **kwargs)
        
 
    
    #===========================================================================
    # PLOTRS------
    #===========================================================================
 
def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r0',
        overwrite=True,
        
        #=======================================================================
        # data files
        #=======================================================================
        catalog_fp='',
        
        #=======================================================================
        # parameters
        #=======================================================================
 
        #=======================================================================
        # plot control
        #=======================================================================
        transparent=False,
        
        #=======================================================================
        # debugging
        #=======================================================================
 
        
        **kwargs):
    
    with ExpoPlotr(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                        
                       bk_lib = {
                           'catalog':dict(catalog_fp=catalog_fp),
                           
                           },
                 **kwargs) as ses:
        
        #=======================================================================
        # compiling-----
        #=======================================================================
 
        ses.compileAnalysis()
 
        #=======================================================================
        # PLOTS------
        #=======================================================================
        #change order
        dx_raw = ses.retrieve('catalog').loc[idx[('obwb', 'LMFRA', 'Calgary', 'noise'), :], :]
        """
        view(dx_raw)
        """
 
        #resolution filtering
        hi_res=10**3 #max we're using for hyd is 300
        hr_dx = dx_raw.loc[dx_raw.index.get_level_values('resolution')<=hi_res, :]
        
 
        
        figsize=(8,12)
        for plotName, dx, xlims,  ylims,xscale, yscale, drop_zeros in [
            ('',dx_raw, None,None, 'log', 'linear', True),
 
            #('hi_res',hr_dx, (10, hi_res),None, 'linear', 'linear', True),
            #('hi_res_2',hr_dx, (10, hi_res),(-0.1,1), 'linear', 'linear', False),
            ]:
            print('\n\n%s\n\n'%plotName)
 
 
            

        
        out_dir = ses.out_dir
    return out_dir

 
 

def r01():
    return run(tag='r01',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydR01\hydR01_run_index.csv',)

if __name__ == "__main__": 
    #wet mean

    r01()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    