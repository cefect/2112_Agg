'''
Created on Sep. 10, 2022

@author: cefect
'''
import numpy as np
import numpy.ma as ma
import pandas as pd
import os, copy, datetime
idx= pd.IndexSlice


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

#from agg2.haz.da import UpsampleDASession
from agg2.expo.scripts import ExpoSession
from hp.plot import Plotr, view


def now():
    return datetime.datetime.now()


class ExpoDASession(ExpoSession, Plotr):
    
    def join_arsc_stats(self,fp_lib,
                         **kwargs):
        """stats on resample class of assets"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('arsc_stats',  subdir=True,ext='.pkl', **kwargs)
        
        res_d = dict()
        for method, d1 in fp_lib.items():
            d = dict()
            for base, fp in d1.items():
        
                dx_raw = pd.read_pickle(fp)
                
                d[base] = dx_raw.sum(axis=0).unstack('dsc')
                
            #wrap method
            res_d[method] = pd.concat(d, axis=1, names=['base'])
            
        #wrap
        pd.concat(res_d)