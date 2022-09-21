'''
Created on Sep. 10, 2022

@author: cefect
'''
import os, copy, datetime, logging
import numpy as np
import numpy.ma as ma
import pandas as pd
from pandas.testing import assert_index_equal

idx= pd.IndexSlice

#from definitions import max_cores
#from multiprocessing import Pool

#from agg2.haz.coms import coldx_d, cm_int_d
from hp.pd import append_levels
from hp.basic import lib_iter
 
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

#from agg2.haz.da import UpsampleDASession
from agg2.expo.scripts import ExpoSession
from agg2.coms import Agg2DAComs
from hp.plot import view


def now():
    return datetime.datetime.now()


class ExpoDASession(ExpoSession, Agg2DAComs):
 
    def __init__(self,scen_name='expo_da',  **kwargs):
 
 
 
        super().__init__(scen_name=scen_name, logfile_duplicate=False,
                          **kwargs)
 
    
    def join_layer_samps(self,fp_lib,
                         dsc_df=None,
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
                """
                pd.read_pickle(fp).hist()
                """
                
            #wrap method
            res_d[method] = pd.concat(d, axis=1).droplevel(0, axis=1) #already a dx
            
        #wrap
        dx1 =  pd.concat(res_d, axis=1, names=['method']).sort_index(axis=1)
        
 
            
 
        
        return dx1
    
    def get_dsc_stats2(self, raw_dx,
                       ufunc_d = {'expo':'sum', 'wd':'mean', 'wse':'mean'},
                       **kwargs):
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dscStats',  subdir=True,ext='.pkl', **kwargs)
        
        gcols = ['method', 'layer']
        res_d = dict()
        for i, (gkeys, gdx) in enumerate(raw_dx.groupby(level=gcols, axis=1)):
            gkeys_d = dict(zip(gcols, gkeys))
            stat = ufunc_d[gkeys_d['layer']]
            
            #get zonal stats            
            grouper = gdx.groupby(level='dsc')            
            sdx = getattr(grouper, stat)()
            
            #get total stat
            sdx.loc['full', :] = getattr(gdx, stat)()
            
            res_d[i] = pd.concat({stat:sdx}, axis=1, names=['metric'])
 
        return pd.concat(list(res_d.values()), axis=1).reorder_levels(list(raw_dx.columns.names) + ['metric'], axis=1)
    
 
    def plot_grid_d(self, 
                    data_lib,post_lib,
                    title=None, colorMap=None, color_d=None,
                    matrix_kwargs=dict(figsize=(17*cm, 18*cm) , set_ax_title=False, add_subfigLabel=True),
                    plot_kwargs_lib={
                                          'full':{'marker':'x'},
                                          'DD':{'marker':'s', 'fillstyle':'none'},
                                          'WW':{'marker':'o', 'fillstyle':'full'},
                                          'WP':{'marker':'o', 'fillstyle':'top'},
                                          'DP':{'marker':'o', 'fillstyle':'bottom'},
                                          },
                    plot_kwargs={'linestyle':'solid', 'marker':'x', 'markersize':7, 'alpha':0.8},
                    output_fig_kwargs=dict(),
                    **kwargs):
        """grid plot from data in a dict. save post for caller"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('plot_gd',  subdir=False,ext='.svg', **kwargs)
        
        #=======================================================================
        # extract
        #=======================================================================
        # get first frame
        df0 = next(iter(next(iter(data_lib.values())).values()))
        
        keys_all_d = {'row':list(data_lib.keys()),
                      'col':list(next(iter(data_lib.values())).keys()),  # taking from first
                      'color':df0.columns.values.tolist(),
                      }
        
        # color
        color_key = df0.columns.name
        if color_d is None:
            color_d = self._get_color_d(color_key, keys_all_d['color'], colorMap=colorMap, color_d=color_d)
        
        # plot kwargs
        """here we populate with blank kwargs to ensure every series has some kwargs"""
        if plot_kwargs_lib is None: plot_kwargs_lib = dict()
        for k in keys_all_d['color']:
            if not k in plot_kwargs_lib:
                plot_kwargs_lib[k] = plot_kwargs
            else:
                plot_kwargs_lib[k] = {**plot_kwargs, **plot_kwargs_lib[k]}  # respects precedent
        
        log.info('plotting\n    rows:%s\n    cols:%s' % (keys_all_d['row'], keys_all_d['col']))
        #=======================================================================
        # setup figure
        #=======================================================================
        plt.close('all')
 
        fig, ax_d = self.get_matrix_fig(keys_all_d['row'], keys_all_d['col'],
                                    # figsize_scaler=4,                                    
                                    constrained_layout=False,
                                    sharey='row',sharex='all',  
                                    fig_id=0,logger=log, **matrix_kwargs)
 
        if not title is None:
            fig.suptitle(title)
            
        #=======================================================================
        # loop and plot
        #=======================================================================

        #loop over the nested dictionary
        cnt=0
        for row_key, col_key, df in lib_iter(data_lib):
 
            #===================================================================
            # defaults
            #===================================================================
            log.debug(f'    on {row_key}.{col_key} for {str(df.shape)}')
            ax = ax_d[row_key][col_key]
            
            #check
            assert isinstance(df, pd.DataFrame)
            assert isinstance(df.columns, pd.Index)
            
            #===================================================================
            # plot each series (diff colors)
            #===================================================================
            for col_lab, col in df.items():
                ax.plot(col.index, col.values, color=color_d[col_lab],label=col_lab,**plot_kwargs_lib[col_lab])
            cnt+=1
        
        log.info('built %i plots'%cnt)
 
            
        
        return ax_d, keys_all_d
    
    
        

    
    
    
    
