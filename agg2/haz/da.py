'''
Created on Aug. 30, 2022

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

from agg2.haz.scripts import UpsampleSession, assert_dx_names
from agg2.coms import Agg2DAComs
from hp.plot import view


def now():
    return datetime.datetime.now()


class UpsampleDASession(Agg2DAComs, UpsampleSession):
    """dataanalysis of downsampling"""

    def __init__(self,  obj_name='ups', **kwargs):
 
 
 
        super().__init__(obj_name=obj_name, **kwargs)
 
        
    def join_stats(self,fp_lib, **kwargs):
        """merge results from run_stats for different methodss and clean up the data"""
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('jstats',  subdir=False,ext='.xls', **kwargs)
        
        #=======================================================================
        # preckec
        #=======================================================================
        for k1, d in fp_lib.items():
            for k2, fp in d.items():
                assert os.path.exists(fp), '%s.%s' % (k1, k2)
        
        #=======================================================================
        # loop and join
        #=======================================================================
        res_lib = dict()
        for k1, fp_d in fp_lib.items():
            res_d = dict() 
            for k2, fp in fp_d.items():
                
                dxcol_raw = pd.read_pickle(fp)            
                log.info('for %s.%s loading %s' % (k1, k2, str(dxcol_raw.shape)))
                
                # check
                assert_dx_names(dxcol_raw, msg='%s.%s' % (k1, k2))
                
                res_d[k2] = dxcol_raw
        
            #===================================================================
            # wrap reference
            #===================================================================
            res_lib[k1] = pd.concat(res_d, axis=1, names=['base'])            
        
        #=======================================================================
        # #concat
        #=======================================================================
        rdxcol = pd.concat(res_lib, axis=1, names=['method']
                   ).swaplevel('base', 'method', axis=1).sort_index(axis=1).sort_index(axis=0)
        
        #=======================================================================
        # #relabel all
        #=======================================================================
        idf = rdxcol.columns.to_frame().reset_index(drop=True)
        idf.loc[:, 'dsc'] = idf['dsc'].replace({'all':'full'})
        rdxcol.columns = pd.MultiIndex.from_frame(idf)
        
        #=======================================================================
        # write
        #=======================================================================
        if write:
            with pd.ExcelWriter(ofp, engine='xlsxwriter') as writer: 
                rdxcol.to_excel(writer, sheet_name='stats', index=True, header=True)
            log.info('wrote %s to \n    %s' % (str(rdxcol.shape), ofp))
        #=======================================================================
        # wrap
        #=======================================================================
        metric_l = rdxcol.columns.get_level_values('metric').unique().to_list()
        log.info('finished on %s w/ %i metrics \n    %s' % (str(rdxcol.shape), len(metric_l), metric_l))
        
        return rdxcol
    
    def get_s12N(self,dx_raw,
                 ):
        
        """probably some way to do this natively w/ panda (transform?)
        but couldnt figure out how to divide across 2 levels
        
        made a functio nto workaround the access violation
        """
        
        base_dxcol = dx_raw.loc[1, idx['s1', 'direct',:, 'full',:]].droplevel((0, 1, 3), axis=1).reset_index(drop=True)  # baseline values
        
        d = dict()
        for layName, gdx in dx_raw['s12'].groupby('layer', axis=1):
            base_ser = base_dxcol[layName].iloc[0,:]
            d[layName] = gdx.droplevel('layer', axis=1).divide(base_ser, axis=1, level='metric')
            
        #concat and promote
        div_dxcol = pd.concat(d, axis=1, names=['layer'])
        return pd.concat([div_dxcol], names=['base'], keys=['s12N'], axis=1).reorder_levels(dx_raw.columns.names, axis=1)
    
        """
        view(rdxcol)
        """
 
        
    

 

        
        
 

    
    
    
    
    
    
    
    
    
    
    
    
