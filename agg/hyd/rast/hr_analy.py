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

from agg.hyd.rast.hr_scripts import RastRun, view
from hp.plot import Plotr

class RasterAnalysis(RastRun, Plotr): #analysis of model results
    resCn='resolution'
    saCn='studyArea'
    
        #colormap per data type
    colorMap_d = {
 
        'studyArea':'Dark2',
 
        'resolution':'copper',
 
        }
    
    def __init__(self,
 
                 name='rast_analy',
                 plt=None,
                 exit_summary=False,
                 **kwargs):
        
        data_retrieve_hndls = {
            'rstats':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_rstats(**kwargs), #
                },

            
            
            }
        
        self.plt=plt
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls,name=name,init_plt_d=None,
                         exit_summary=exit_summary,**kwargs)
        
    def runRastAnalysis(self,
                        ):
        self.retrieve('rstats')
        
        
        
    def build_rstats(self, #just combing all the results
                     dkey='rstats',
                     logger=None,write=None,
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_rstats')
        if write is None: write=self.write
        assert dkey=='rstats'
        
        #=======================================================================
        # retrieve from hr_scripts
        #=======================================================================
        dxb = self.retrieve('rstats_basic')
        
        dxw = self.retrieve('wetAreas')
        
        assert_index_equal(dxb.index, dxw.index)
        
        assert np.array_equal(dxb.index.names, np.array([self.resCn, self.saCn]))
        
        #=======================================================================
        # join
        #=======================================================================
        rdx = dxb.join(dxw).sort_index()
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(rdx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return rdx
    
    def plot_progression(self,
                         #data
                         dx_raw=None, #combined model results
                         coln = 'MEAN', #variable to plot against resolution
                         
                         #plot control
                         plot_colr=None,
                         ax=None,figsize=(6.5,4), colorMap=None,
                         plot_kwargs = dict(marker='x'),
                         title=None,xlabel=None,ylabel=None,
                         xscale='log',
                         xlims=None,
                         
                         logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is  None: logger=self.logger
        log = logger.getChild('plot_progression')
        resCn, saCn = self.resCn, self.saCn
        
        if plot_colr is None: plot_colr=saCn
        if colorMap is None: colorMap=self.colorMap_d[plot_colr]
        
        if xlabel is None: xlabel = resCn
        if ylabel is None: ylabel = coln
        #=======================================================================
        # retrival
        #=======================================================================
        if dx_raw is None: dx_raw = self.retrieve('rstats')
        """
        view(dx_raw)
        """
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert coln in dx_raw.columns
        
        #=======================================================================
        # data prep
        #=======================================================================
        log.info('on %i'%len(dx_raw))
        serx = dx_raw[coln]
        mdex = serx.index
        #=======================================================================
        # setup axis
        #=======================================================================
        if ax is None:
            fig = plt.figure(figsize=figsize,
                     tight_layout=False,
                     constrained_layout = True,
                     )
            
            ax = fig.add_subplot()
            
        else:
            fig = ax.figure
            
        """
        fig.show()
        """
        #title
        if title is None:
 
            title='%s vs. %s'%(coln, resCn)
        #get colors
        ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
            
        #=======================================================================
        # loop and plot
        #=======================================================================
        gcols = [saCn]
        for gkeys, gsx0 in serx.groupby(level=gcols):
            if isinstance(gkeys, str): gkeys=[gkeys]
            keys_d = dict(zip(gcols, gkeys))
            log.info('on %s w/ %i'%(keys_d, len(gsx0)))
            
            #===================================================================
            # data prep
            #===================================================================
            xar = gsx0.index.get_level_values(resCn).values #resolutions
            yar = gsx0.values
            color=color_d[keys_d[plot_colr]]
            #===================================================================
            # plot
            #===================================================================
            
            ax.plot(xar, yar, color=color,label =''.join(gkeys), **plot_kwargs)
            
            
        #===================================================================
        # post format
        #===================================================================
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()
        
        #chang eto log scale
        ax.set_xscale(xscale)
        
        
        if not xlims is None:
            ax.set_xlim(xlims)
        

        
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
 
                           
                           },
                 **kwargs) as ses:
        
        #=======================================================================
        # compiling-----
        #=======================================================================
        #ses.runRastAnalysis()
        
        #=======================================================================
        # dx_raw = ses.retrieve('rstats')
        # bx = np.logical_and(
        #     dx_raw.index.get_level_values('studyArea')=='LMFRA',
        #     dx_raw.index.get_level_values('resolution')<=1280,
        #     )
        # dx = dx_raw.loc[bx, :]
        # 
        #=======================================================================
        dx=None
        xlims = None
        
        ses.plot_progression(coln='MEAN', ylabel='depth (m)', dx_raw=dx, xlims=xlims)
        
        ses.plot_progression(coln='wetAreas', ylabel='inundation area (m2)', dx_raw=dx, xlims=xlims)
        
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
        
    
    
if __name__ == "__main__": 
    
    #dev()
    r1()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    