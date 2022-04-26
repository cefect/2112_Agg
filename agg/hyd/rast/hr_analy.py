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
    def __init__(self,
 
                 name='rastAnaly',
                 plt=None,
                 exit_summary=False,
                 **kwargs):
        
        data_retrieve_hndls = {
            'rstats':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_rstats(**kwargs), #
                },

            
            
            }
        
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
                         ax=None,
                         logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is  None: logger=self.logger
        log = logger.getChild('plot_progression')
        
        #=======================================================================
        # retrival
        #=======================================================================
        if dx_raw is None: dx_raw = self.retrieve('rstats')
        """
        view(dx_raw)
        """
        
        log.info('on %i'%len(dx_raw))
        
        #=======================================================================
        # setup axis
        #=======================================================================
        
    
    

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
        
        ses.plot_progression()
        
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
    
    
if __name__ == "__main__": 
    
    dev()
    pass

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    