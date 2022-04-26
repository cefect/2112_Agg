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
        'size'   : 8}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 10 #set the figure title size
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)

from agg.hyd.rast.hrast import RastRun
from hp.plot import Plotr

class RasterAnalysis(HydSession, Plotr): #analysis of model results
    def __init__(self,
                 catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hyd2\model_run_index.csv',
                 plt=None,
                 name='analy',
                 modelID_l=None, #optional list for specifying a subset of the model runs
                 #baseID=0, #mase model ID
                 exit_summary=False,
                 **kwargs):
        
        data_retrieve_hndls = {
            'r':{
                #probably best not to have a compliled version of this
                'build':lambda **kwargs:self.build_catalog(**kwargs), #
                },

            
            
            }
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls,name=name,init_plt_d=None,
                         work_dir = r'C:\LS\10_OUT\2112_Agg', exit_summary=exit_summary,
                         **kwargs)