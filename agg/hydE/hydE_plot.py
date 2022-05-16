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
        'size'   : 12}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 14 
matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)

from agg.hydR.hydR_plot import RasterPlotr
from agg.hydE.hydE_scripts import ExpoRun
from hp.plot import Plotr
from hp.gdal import rlay_to_array, getRasterMetadata
from hp.basic import set_info, get_dict_str
from hp.pd import get_bx_multiVal, view
#from hp.animation import capture_images

class ExpoPlotr(RasterPlotr, ExpoRun): #analysis of model results

    
    def __init__(self,
 
                 name='hydE_plot',
                 colorMap_d={
                     
                     },
 
                 **kwargs):
        
        
        colorMap_d.update({
            'aggLevel':'cool'
                        })
 
        
        super().__init__(name=name,colorMap_d=colorMap_d, **kwargs)
        
 
 
    
    #===========================================================================
    # PLOTRS------
    #===========================================================================
 
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
    
    with ExpoPlotr(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                        
                       bk_lib = {
                           'catalog':dict(catalog_fp=catalog_fp),
                           
                           },
                 **kwargs) as ses:
        
        #=======================================================================
        # compiling-----
        #=======================================================================
 
        ses.compileAnalysis()
 
        #=======================================================================
        # PLOTS------
        #=======================================================================
        #change order
        dx_raw = ses.retrieve('catalog').loc[idx[('obwb', 'LMFRA', 'Calgary', 'noise'), :], :]
        
        #just the expo stats
        dx1=dx_raw
        #dx1 = dx_raw.loc[:, idx[('rsampStats'), :]].droplevel(0, axis=1)
        """
        view(dx_raw)
        """
 
        #resolution filtering
        hi_res=10**3 #max we're using for hyd is 300
        hr_dx = dx1.loc[dx_raw.index.get_level_values('resolution')<=hi_res, :]
        
 
        
        figsize=(8,12)
        for plotName, dxi, xlims,  ylims,xscale, yscale, drop_zeros in [
            ('',dx1, None,None, 'log', 'linear', True),
 
            #('hi_res',hr_dx, (10, hi_res),None, 'linear', 'linear', True),
            #('hi_res_2',hr_dx, (10, hi_res),(-0.1,1), 'linear', 'linear', False),
            ]:
            print('\n\n%s\n\n'%plotName)
            
            #=======================================================================
            # multi-metric vs Resolution---------
            #=======================================================================
            #nice plot showing the major raw statistics 


     
            #===================================================================
            # compare dsampStage
            #===================================================================
            bx = dxi.index.get_level_values('downSampling')=='Average'
            
 

            """
            view(dx)
            """
            #===================================================================
            # ses.plot_StatXVsResolution(
            #     set_ax_title=False,
            #     dx_raw=dx[bx].droplevel(0, axis=1), 
            #     coln_l=list(col_d.keys()), xlims=xlims,ylab_l = list(col_d.values()),
            #     title=plotName)
            #===================================================================
            
            #===================================================================
            # compare downSampling
            #===================================================================
            print('\n\n %s: downSampling comparison\n\n'%plotName)
            
            for dsampStage in [
                #'postFN', 
                'pre']:
                bx = dxi.index.get_level_values('dsampStage')==dsampStage
                assert bx.any()
    
                """
                view(dxi)
                """
                #===============================================================
                # plot exposure
                #===============================================================
                col_d={#exposure values
                        'mean': 'sampled mean (m)',
                        #'max': 'sampled max (m)',
                        #'min': 'sampled min (m)',
                        'wet_mean': 'sampled wet mean (m)',
                        #'wet_max': 'sampled wet max (m)',
                        #'wet_min': 'sampled wet min (m)',
                        'wet_pct': 'wet assets (pct)'
                        }
                
                ax_d = ses.plot_StatXVsResolution(
                    set_ax_title=False,xvar='aggLevel',
                    dx_raw=dxi[bx].loc[:, idx[('rsampStats'), :]].droplevel(0, axis=1), 
                    coln_l=list(col_d.keys()), xlims=xlims,ylab_l = list(col_d.values()),
                    title=plotName + '_'+dsampStage, plot_bgrp='resolution',
                    write=True)
                
                break
                                
                            
                ax_d = ses.plot_StatXVsResolution(
                    set_ax_title=False,
                    dx_raw=dxi[bx].loc[:, idx[('rsampStats'), :]].droplevel(0, axis=1), 
                    coln_l=list(col_d.keys()), xlims=xlims,ylab_l = list(col_d.values()),
                    title=plotName + '_'+dsampStage, plot_bgrp='aggLevel',
                    write=False)
                
                #===============================================================
                # add raster
                #===============================================================
                """order matters here
                we plot teh new data on whatever axis order is specified above"""
                col_d1={#raster values
                        #===============================================================
                        # 'MAX':'max depth (m)',
                        # 'MIN':'min depth (m)',
                         'MEAN':'global mean depth (m)',
 
                        'wetMean':'wet mean depth (m)',
                        #'wetVolume':'wet volume (m^3)', 
                         #'wetArea': 'wet area (m^2)', 
                         #'rmse':'RMSE (m)',
                         #'gwArea':'gwArea',
                         #'STD_DEV':'stdev (m)',
                         #'noData_cnt':'noData_cnt',
                         'noData_pct':'no data (%)'
                    
                          }
                
                bx1 = np.logical_and(bx,dxi.index.get_level_values('aggLevel')==1) #only 1 agg level
                dxi1=dxi[bx1].loc[:, idx[('rstats', 'wetStats', 'noData_pct'), :]].droplevel(0, axis=1)
                                       
                plot_colr='dsampStage'
                ses.plot_StatXVsResolution(ax_d=ax_d,
 
                    plot_colr=plot_colr,
                   color_d={k:'black' for k in dxi1.index.unique(plot_colr)},
                   marker_d={k:'x' for k in dxi1.index.unique(plot_colr)},
                   plot_kwargs = dict(alpha=0.8, linestyle='dashed'),
                                           
                    set_ax_title=False,
                    dx_raw=dxi1, 
                    coln_l=list(col_d1.keys()), 
                    #xlims=xlims,
                    ylab_l = list(col_d.values()),
                    title=plotName + '_'+dsampStage )
 
            

        
        out_dir = ses.out_dir
    return out_dir

 
 

def r01():
    return run(tag='r01',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydE01\hydE01_run_index.csv',)

if __name__ == "__main__": 
    #wet mean

    r01()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    