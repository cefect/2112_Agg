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
from hp.plot import Plotr, view


def now():
    return datetime.datetime.now()


class UpsampleDASession(UpsampleSession, Plotr):
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
 
        
    
    def plot_matrix_metric_method_var(self,
                                      serx,
                                      map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'pixelLength'},
                                      title=None, colorMap=None,color_d=None,
                                      ylab_d={'vol':'$V_{s2}$ (m3)', 'wd_mean':r'$WD_{s2}$ (m)', 'wse_area':'$A_{s2}$ (m2)'},
                                      ax_title_d={'direct':'direct', 'filter':'filter and subtract'},
                                      xscale='linear',
                                      matrix_kwargs = dict(figsize=(6.5,6)),
                                      plot_kwargs_lib={
                                          'full':{'marker':'x'},
                                          'DD':{'marker':'s', 'fillstyle':'none'},
                                          'WW':{'marker':'o', 'fillstyle':'full'},
                                          'WP':{'marker':'o', 'fillstyle':'top'},
                                          'DP':{'marker':'o', 'fillstyle':'bottom'},
                                          },
                                      plot_kwargs={'linestyle':'solid', 'marker':'x', 'markersize':7, 'alpha':0.8}, 
 
                                      **kwargs):
        
        """build matrix plot of variance
            x:pixelLength
            y:(series values)
            rows: key metrics (wd_mean, wse_area, vol)
            cols: all methods
            colors: downsample class (dsc)
            
        Parameters
        -----------
        serx: pd.Series w/ multindex
            see join_stats
        map_d: dict
            plot matrix name dict mapping dxcol data labels to the matrix plot
            
        plot_kwargs_lib: {series name: **plot_kwargs}
            series specific kwargs
        
        plot_kwargs: dict
            kwargs for all series (gives up precedent to series specific)
            
        Note
        --------
        cleaner to do all slicing and data maniupation before the plotter
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('metric_method_var',  subdir=False,ext='.svg', **kwargs)
 
            
        #=======================================================================
        # extract data
        #=======================================================================
        map_d = {k:map_d[k] for k in ['row', 'col', 'color', 'x']} #ensure order on map
        
        """leaving data order passed by theuser"""
        serx = serx.reorder_levels(list(map_d.values()))#.sort_index(level=map_d['x']) #ensure order on data
        
        mdex = serx.index
        keys_all_d = {k:mdex.unique(v).tolist() for k,v in map_d.items()} #order matters
        
        if color_d is None:
            color_d = self._get_color_d(map_d['color'], keys_all_d['color'], colorMap=colorMap, color_d=color_d)
        
        #plot kwargs
        """here we populate with blank kwargs to ensure every series has some kwargs"""
        if plot_kwargs_lib is None: plot_kwargs_lib=dict()
        for k in keys_all_d['color']:
            if not k in plot_kwargs_lib:
                plot_kwargs_lib[k] = plot_kwargs
            else:
                plot_kwargs_lib[k] = {**plot_kwargs, **plot_kwargs_lib[k]} #respects precedent
 
        #=======================================================================
        # setup figure
        #=======================================================================
        plt.close('all')
 
 
        fig, ax_d = self.get_matrix_fig(keys_all_d['row'], keys_all_d['col'],
                                    #figsize_scaler=4,                                    
                                    constrained_layout=True,
                                    sharey='row',sharex='all',  
                                    fig_id=0,
                                    set_ax_title=False, add_subfigLabel=True,
                                    **matrix_kwargs)
     
 
        if not title is None:
            fig.suptitle(title)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        levels = [map_d[k] for k in ['row', 'col']]
        for gk0, gsx0 in serx.groupby(level=levels):
            #===================================================================
            # setup
            #===================================================================
            ax = ax_d[gk0[0]][gk0[1]]
            keys_d = dict(zip(levels, gk0))
            
            ax.set_xscale(xscale)
            #===================================================================
            # loop each series (color group)
            #===================================================================
            for gk1, gsx1 in gsx0.groupby(level=map_d['color']):
                keys_d[map_d['color']] = gk1
                xar, yar = gsx1.index.get_level_values(map_d['x']).values, gsx1.values
                #===============================================================
                # plot
                #===============================================================
                ax.plot(xar, yar, color=color_d[gk1],label=gk1,**plot_kwargs_lib[gk1])
                
        #=======================================================================
        # post format
        #=======================================================================
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                ax.grid()
                
                #first row
                if row_key==keys_all_d['row'][0]:
                    ax.set_title(ax_title_d[col_key])
                    
                    
                #first col
                if col_key == keys_all_d['col'][0]:
                    ax.set_ylabel(ylab_d[row_key])
                    
                    #force 2decimal precision
                    ax.get_yaxis().set_major_formatter(lambda x,p:'%.2f'%x)
                    
                
                #last row
                if row_key==keys_all_d['row'][-1]:
                    ax.set_xlabel('resolution (m)')
                
                #last col
                if col_key == keys_all_d['col'][-1]:
                    
                    #first row
                    if row_key==keys_all_d['row'][0]:
                        ax.legend()
                        
        #=======================================================================
        # output
        #=======================================================================
        return self.output_fig(fig, ofp=ofp, logger=log)
    
    def plot_dsc_ratios(self, df,
                        colorMap=None,color_d=None,
                        **kwargs):
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('dsc_rats',  subdir=False,ext='.svg', **kwargs)
        
        #=======================================================================
        # setup
        #=======================================================================
 
        coln = df.columns.name
        keys_all_d={'color':df.columns.tolist()}
        
        if color_d is None:
            color_d = self._get_color_d(coln, keys_all_d['color'], colorMap=colorMap, color_d=color_d)
        
        color_l = [color_d[k] for k in df.columns]
            
        #=======================================================================
        # setup plot
        #=======================================================================
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6.5,2), constrained_layout=True)
            
        #=======================================================================
        # loop and plot
        #=======================================================================
        ax.stackplot(df.index, df.T.values, labels=df.columns, colors=color_l,
                     alpha=0.8)
 
        
        #=======================================================================
        # format
        #=======================================================================
        ax.legend(loc=3)
        ax.set_xlabel('pixel size (m)')
        ax.set_ylabel('domain fraction')
        ax.set_ylim((0.6,1.0))
        
        #=======================================================================
        # output
        #=======================================================================
        return self.output_fig(fig, ofp=ofp, logger=log)
 

        
        
 

    
    
    
    
    
    
    
    
    
    
    
    