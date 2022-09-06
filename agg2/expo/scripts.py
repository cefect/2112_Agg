'''
Created on Sep. 6, 2022

@author: cefect

aggregation exposure/assetse
'''
#===============================================================================
# IMPORTS-------
#===============================================================================
from definitions import wrk_dir
import numpy as np
import pandas as pd
idx= pd.IndexSlice
import os, copy, datetime

import geopandas as gpd
import rasterio as rio
from rasterstats import zonal_stats
import rasterstats.utils
import matplotlib.pyplot as plt


from hp.oop import Session
from hp.gpd import GeoPandasWrkr, assert_intersect, ds_get_bounds
from hp.rio import load_array, RioWrkr
from hp.pd import view
 
from agg2.haz.rsc.scripts import ResampClassifier

def now():
    return datetime.datetime.now()


class ExpoSession(GeoPandasWrkr, ResampClassifier, Session):
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
                       bbox=None,
                        **kwargs):
        """join resampClass to each asset (one column per resolution)"""
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('arsc',  subdir=True,ext='.pkl', **kwargs)
        
        if bbox is None: bbox=self.bbox
        
        #=======================================================================
        # classification masks
        #=======================================================================
        df_raw = pd.read_pickle(pick_fp).loc[:, ['downscale', 'catMosaic_fp']]
        cm_fp_d = df_raw.set_index('downscale').dropna().iloc[:,0].to_dict()  
        
        """
        view(pd.read_pickle(pick_fp))
        """ 
        
        log.info('on %i catMasks'%len(cm_fp_d))
        
        for k,v in cm_fp_d.items():
            assert os.path.exists(v), k
        #=======================================================================
        # load asset data         
        #=======================================================================
        gdf = gpd.read_file(finv_fp, bbox=bbox)
        assert gdf.crs==self.crs, 'crs mismatch'
        #=======================================================================
        # loop downscales
        #=======================================================================
        res_d = 
        for scale, rlay_fp in cm_fp_d.items():
            log.info('on %i w/ %s'%(scale, os.path.basename(rlay_fp)))
            with rio.open(rlay_fp, mode='r') as ds:
                
                #check consistency
                assert ds.crs.to_epsg()==self.crs.to_epsg()
                assert_intersect(ds_get_bounds(ds).bounds, tuple(gdf.total_bounds.tolist()))
                
                #===============================================================
                # loop each category's mask'
                #===============================================================
                mosaic_ar = load_array(ds) 
                cm_d = self.mosaic_to_masks(mosaic_ar)
                
                #load and compute zonal stats
                zd = dict()
                for catid, ar_raw in cm_d.items():
                    
                    if np.any(ar_raw):
                        stats_d = zonal_stats(gdf, np.where(ar_raw, 1, 0), 
                                    affine=ds.transform,
                                    nodata=0,
                                    all_touched=False,
                                    stats=[
                                            #'count', 
                                           'mean'
                                           ],
                                    
                                    )
                    
                        zd[catid] = pd.DataFrame(stats_d)
                #===============================================================
                # wrap
                #===============================================================
                pd.concat(zd, axis=1).droplevel(1, axis=1)
                raise IOError('stopped here')
                
 
        
        
        """
        gdf.plot()
        plt.show()
        """
        
        
        
        
        
    
        
 