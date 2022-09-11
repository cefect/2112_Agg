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


class UpsampleDASession(UpsampleSession, Agg2DAComs):
    """dataanalysis of downsampling"""
    colorMap_d = {
        'dsc':'PiYG'
        }
    
    color_lib = {
        'dsc':{  
                'WW':'#0000ff',
                'WP':'#00ffff',
                'DP':'#ff6400',
                'DD':'#800000',
                'full': '#000000'}
        }
    
 
        
    def join_stats(self,fp_lib, **kwargs):
        """merge results from run_stats for different methodss and clean up the data"""
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('jstats',  subdir=False,ext='.xls', **kwargs)
        
        #=======================================================================
        # preckec
        #=======================================================================
        for k1,d in fp_lib.items():
            for k2, fp in d.items():
                assert os.path.exists(fp), '%s.%s'%(k1, k2)
        
        #=======================================================================
        # loop and join
        #=======================================================================
        res_lib = dict()
        for k1,fp_d in fp_lib.items():
            res_d = dict() 
            for k2, fp in fp_d.items():
                
                dxcol_raw = pd.read_pickle(fp)            
                log.info('for %s.%s loading %s'%(k1, k2, str(dxcol_raw.shape)))
                
                #check
                assert_dx_names(dxcol_raw, msg='%s.%s'%(k1, k2))
                
                res_d[k2] = dxcol_raw
 
                #===============================================================
                # #drop excess levels
                #===============================================================
                #===============================================================
                # if len(dxcol_raw.index.names)>1:
                #     #retrieve hte meta info
                #     meta_df = dxcol_raw.index.to_frame().reset_index(drop=True).set_index('scale').sort_index(axis=0)
                #       
                #     #remove from index
                #     dxcol = dxcol_raw.droplevel((1,2))
                # else:
                #     dxcol = dxcol_raw
                #===============================================================
                    
                
                #===============================================================
                # #append levels
                #===============================================================
                #===============================================================
                # dx1 = pd.concat({k1:pd.concat({k2:dxcol}, names=['metricLevel'], axis=1)},
                #           names=['method'], axis=1)
                # 
                # #===============================================================
                # # add a dummy for missings
                # #===============================================================
                # miss_l = set(rdx.columns.names).symmetric
                # #===============================================================
                # # start
                # #===============================================================
                # if rdx is None:
                #     rdx = dx1.copy()
                #     continue
                # 
                # try:
                #     rdx = rdx.join(dx1)
                # except Exception as e:
                #     """
                #     view(dx1)
                #     """
                #     raise IndexError('failed to join %s.%s. w/ \n    %s'%(k1, k2, e))
                #===============================================================
        
            #===================================================================
            # wrap reference
            #===================================================================
            res_lib[k1] = pd.concat(res_d, axis=1, names=['base'])            
        
        #=======================================================================
        # #concat
        #=======================================================================
        rdxcol = pd.concat(res_lib, axis=1,  names=['method']
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
            log.info('wrote %s to \n    %s'%(str(rdxcol.shape), ofp))
        #=======================================================================
        # wrap
        #=======================================================================
        metric_l = rdxcol.columns.get_level_values('metric').unique().to_list()
        log.info('finished on %s w/ %i metrics \n    %s'%(str(rdxcol.shape), len(metric_l), metric_l))
        

        
        return rdxcol
    
        """
        view(rdxcol)
        """
 
        
    

 

        
        
 

    
    
    
    
    
    
    
    
    
    
    
    