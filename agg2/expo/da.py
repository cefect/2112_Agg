'''
Created on Sep. 10, 2022

@author: cefect
'''
import numpy as np
import numpy.ma as ma
import pandas as pd
import os, copy, datetime
idx= pd.IndexSlice
from agg2.haz.coms import coldx_d

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
from agg2.coms import Agg2DAComs
from hp.plot import view


def now():
    return datetime.datetime.now()


class ExpoDASession(ExpoSession, Agg2DAComs):
    
    def join_arsc_stats(self,fp_lib,
                         **kwargs):
        """assemble resample class of assets
        
        this is used to tag assets to dsc for reporting.
        can also compute stat counts from this
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('arscJ',  subdir=True,ext='.pkl', **kwargs)
        
        res_d = dict()
        for method, d1 in fp_lib.items():
            d = dict()
            for dsource, fp in d1.items():
                if not dsource=='arsc': continue        
                dx_raw = pd.read_pickle(fp)
                
                d[dsource] = dx_raw
                
            #wrap method
            res_d[method] = pd.concat(d, axis=1, names=['dsource'])
            
        #wrap
        dx1 =  pd.concat(res_d, axis=1, names=['method']).swaplevel('method', 'dsource', axis=1).droplevel('dsource', axis=1).sort_index(axis=1)
 
        
        return dx1
    
    def join_layer_samps(self,fp_lib,
                         **kwargs):
        """assemble resample class of assets
        
        this is used to tag assets to dsc for reporting.
        can also compute stat counts from this
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('lsampsK',  subdir=True,ext='.pkl', **kwargs)
        
        res_d = dict()
        for method, d1 in fp_lib.items():
            d = dict()
            for layName, fp in d1.items():
                if not layName in ['wd', 'wse', 'dem']: continue        
 
                d[layName] = pd.read_pickle(fp)
                
            #wrap method
            res_d[method] = pd.concat(d, axis=1).droplevel(0, axis=1) #already a dx
            
        #wrap
        dx1 =  pd.concat(res_d, axis=1, names=['method']).sort_index(axis=1)
 
        
        return dx1
