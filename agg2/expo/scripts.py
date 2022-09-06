'''
Created on Sep. 6, 2022

@author: cefect

aggregation exposure/assetse
'''
#===============================================================================
# IMPORTS-------
#===============================================================================
import numpy as np
import pandas as pd
import os, copy, datetime
from hp.oop import Session
from hp.gpd import GeoPandasWrkr
import geopandas as gpd
import matplotlib.pyplot as plt
idx= pd.IndexSlice
 
from definitions import wrk_dir
def now():
    return datetime.datetime.now()


class ExpoSession(GeoPandasWrkr, Session):
    """tools for experimenting with downsample sets"""
    
    def __init__(self, 

                 **kwargs):
        """
        
        Parameters
        ----------
 
        """
        
        super().__init__(obj_name='expo', wrk_dir=wrk_dir,subdir=False, **kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
 
        print('finished __init__')
        
        
    def run_expoSubSamp(self):
        """compute resamp class for assets from set of masks"""
        
        #join resampClass to each asset (one column per resolution)
        
        #compute stats
        
    def build_assetRsc(self, pick_fp, finv_fp,
                        **kwargs):
        """join resampClass to each asset (one column per resolution)"""
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('arsc',  subdir=True,ext='.pkl', **kwargs)
        
        
        #=======================================================================
        # classification masks
        #=======================================================================
        df_raw = pd.read_pickle(pick_fp).loc[:, ['downscale', 'catMosaic_fp']]
        
        cM_ser = df_raw.set_index('downscale').dropna().iloc[:,0]       
        
        log.info('on %i catMasks'%len(cM_ser))
        
        #=======================================================================
        # load asset data         
        #=======================================================================
        gdf = gpd.read_file(finv_fp)
        
        gdf.plot()
        """
        plt.show()
        """
        
        
        
        
        
    
        
 