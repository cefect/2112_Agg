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

from agg.hyd.rast.hr_scripts import RastRun, view, Catalog
from hp.plot import Plotr

class RasterAnalysis(RastRun, Plotr): #analysis of model results

    
        #colormap per data type
    colorMap_d = {
 
        'studyArea':'Dark2',
 
        'resolution':'copper',
        'dkey':'Pastel2',
        'dsampStage':'Set1'
 
        }
    
    def __init__(self,
 
                 name='rast_analy',
                 plt=None,
                 exit_summary=False,
                 **kwargs):
        
        data_retrieve_hndls = {
            'catalog':{
                #probably best not to have a compliled version of this
                'build':lambda **kwargs:self.build_catalog(**kwargs), #
                },
            
            }
        
        self.plt=plt
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls,name=name,init_plt_d=None,
                         exit_summary=exit_summary,**kwargs)
        
    def runRastAnalysis(self,
                        ):
        self.retrieve('catalog')
        
        
    def build_catalog(self,
                      dkey='catalog',
                      catalog_fp=None,
                      logger=None,
                      **kwargs):
        if logger is None: logger=self.logger
        assert dkey=='catalog'
        if catalog_fp is None: catalog_fp=self.catalog_fp
        
        return Catalog(catalog_fp=catalog_fp, logger=logger, overwrite=False, **kwargs).get()

    
    def plot_vsResolution(self, #progression against resolution plots
                         #data
                         dx_raw=None, #combined model results
                         coln = 'MEAN', #variable to plot against resolution
                         
                         #plot control
                        plot_type='line', 
                        plot_rown='studyArea',
                        plot_coln=None,
                        plot_colr=None,
                        plot_bgrp='dsampStage', #sub-group onto an axis
                        
                        
                           colorMap=None,
                         plot_kwargs = dict(marker='x'),
                         title=None,xlabel=None,ylabel=None,
                         xscale='log',
                         xlims=None,
                         
                         
                         #plot control [matrix]

                         sharey='none',sharex='col',
                         
                         
                         logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is  None: logger=self.logger
        log = logger.getChild('plot_progression')
        resCn, saCn = self.resCn, self.saCn
        
        if plot_colr is None: plot_colr=plot_bgrp
        if colorMap is None: colorMap=self.colorMap_d[plot_colr]
        
        if xlabel is None: xlabel = resCn
        if ylabel is None: ylabel=coln
        #=======================================================================
        # retrival
        #=======================================================================
        if dx_raw is None: dx_raw = self.retrieve('catalog')
        """
        view(dx_raw)
        dx_raw.index.names
        """
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert coln in dx_raw.columns, coln
 
        
        #=======================================================================
        # data prep
        #=======================================================================
        log.info('on %i'%len(dx_raw))
        serx = dx_raw[coln]
        
        if plot_coln is None:
            """add a dummy level for c onsistent indexing"""
            plot_coln = ''
            serx = pd.concat({plot_coln:serx}, axis=0, names=[plot_coln])
 
        mdex = serx.index
        gcols = set()
        for c in [plot_bgrp, plot_coln, plot_rown]:
            if not c is None: 
                gcols.add(c)
                assert c in mdex.names, c
        gcols = list(gcols)
        #=======================================================================
        # plot defaults
        #=======================================================================
        #title
        if title is None:
            title='%s vs. %s'%(coln, resCn)
        #get colors
        ckeys = mdex.unique(plot_colr)
        
        """nasty workaround to get colors to match w/ hyd""" 
        if plot_colr =='dsampStage':
            ckeys = ['none'] + ckeys.values.tolist()
        
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        if plot_coln is None:
            col_keys = None
        else:
            col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey,sharex=sharex,  
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
 
        fig.suptitle(title)
            
            
            
        """
        fig.show()
        """

            
        #=======================================================================
        # loop and plot
        #=======================================================================
        
        for gkeys, gsx0 in serx.groupby(level=gcols):
            keys_d = dict(zip(gcols, gkeys))
            log.info('on %s w/ %i'%(keys_d, len(gsx0)))
            ax = ax_d[keys_d[plot_rown]][keys_d[plot_coln]]
            #===================================================================
            # data prep
            #===================================================================
            xar = gsx0.index.get_level_values(resCn).values #resolutions
            yar = gsx0.values
            color=color_d[keys_d[plot_colr]]
            #===================================================================
            # plot
            #===================================================================
            if plot_type=='line':
            
                ax.plot(xar, yar, color=color,label =keys_d[plot_colr], **plot_kwargs)
            else:
                raise IOError()
            
            #===================================================================
            # format
            #===================================================================
            #chang eto log scale
            ax.set_xscale(xscale)
            
            if not xlims is None:
                ax.set_xlim(xlims)
            
            
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                # first row
                if row_key == row_keys[0]:
                    #last col
                    if col_key == col_keys[-1]:
                        ax.legend()
                    
                # first col
                if col_key == col_keys[0]:
                    ax.set_ylabel(ylabel)
                
                #last row
                if row_key == row_keys[-1]:
                    ax.set_xlabel(xlabel)
                
                
                ax.grid()
        

        

        
        #=======================================================================
        # wrap
        #=======================================================================
        
        fname = 'progres_%s_%s' % (title,self.longname)
                
        fname = fname.replace('=', '-').replace(' ','').replace('\'','')
        
        
        return self.output_fig(fig, fname=fname)
        
    
    

#===============================================================================
# runners--------
#===============================================================================
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
    
    with RasterAnalysis(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                        
                       bk_lib = {
                           'catalog':dict(catalog_fp=catalog_fp),
                           
                           },
                 **kwargs) as ses:
        
        #=======================================================================
        # compiling-----
        #=======================================================================
        ses.runRastAnalysis()
        
        #=======================================================================
        # PLOTS------
        #=======================================================================
        dx_raw = ses.retrieve('catalog')
        hi_res=350 #max we're using for hyd is 300
        for plotName, dx, xlims, xscale in [
            #('full', dx_raw,(10, 1e4)), 'log', 
            ('hi_res',dx_raw.loc[dx_raw.index.get_level_values('resolution')<=hi_res, :], (10, hi_res), 'linear')
            ]:
            
            coln='MEAN'
            ses.plot_vsResolution(coln=coln, ylabel='mean depth (m)', dx_raw=dx,xlims=xlims,xscale=xscale,
                                  title='%s vs. %s (%s)'%(coln, ses.resCn, plotName))
             
            coln='wetAreas'
            ses.plot_vsResolution(coln=coln, ylabel='wet area (m2)', dx_raw=dx,xlims=xlims,xscale=xscale,
                                  title='%s vs. %s (%s)'%(coln, ses.resCn, plotName))

        
        out_dir = ses.out_dir
    return out_dir

def dev():
    return run(
        tag='dev',
        compiled_fp_d={
            'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\drlay_lib_rast1_dev_0426.pickle',
            'rstats_basic':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\rstats_rast1_dev_0426.pickle',
            'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\wetAreas_rast1_dev_0426.pickle',
            'rstats':r'C:\LS\10_OUT\2112_Agg\outs\rastAnaly\dev\20220426\working\rstats_rastAnaly_dev_0426.pickle',
            }
                
        )
    
def r1():
    return run(
        tag='r1',
 
        compiled_fp_d = {
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\rast\r1\20220426\working\drlay_lib_rast_r1_0426.pickle',
        'rstats_basic':r'C:\LS\10_OUT\2112_Agg\outs\rast\r1\20220426\working\rstats_basic_rast_r1_0426.pickle',
        'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\rast\r1\20220426\working\wetAreas_rast_r1_0426.pickle',
        #'rstats':r'C:\LS\10_OUT\2112_Agg\outs\rast_analy\r1\20220426\working\rstats_rast_analy_r1_0426.pickle',
            }
        )
        
def r2():
    return run(
        tag='r1',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hrast1\hrast_run_index.csv',
        compiled_fp_d = {
 
            }
        )
    
if __name__ == "__main__": 
    
    #dev()
    r2()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    