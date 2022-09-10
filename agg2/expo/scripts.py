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
import shapely.geometry as sgeo
import rasterio as rio
import rasterio.windows

from rasterstats import zonal_stats
import rasterstats.utils
import matplotlib.pyplot as plt


from hp.oop import Session
from hp.gpd import GeoPandasWrkr, assert_intersect
from hp.rio import load_array, RioWrkr, get_window, plot_rast
from hp.pd import view
#from hp.plot import plot_rast
 
from agg2.haz.rsc.scripts import ResampClassifier
from agg2.coms import Agg2Session

def now():
    return datetime.datetime.now()


class ExpoWrkr(GeoPandasWrkr, ResampClassifier):

    def get_assetRsc(self, cm_fp_d, gdf, bbox=None, logger=None):
        """compute zonal stats for assets on each resample class mosaic"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_assetRsc')
        
        #=======================================================================
        # loop and sample
        #=======================================================================
        res_d = dict()
        for scale, rlay_fp in cm_fp_d.items():
            log.info('on scale=%i w/ %s' % (scale, os.path.basename(rlay_fp)))
            with rio.open(rlay_fp, mode='r') as ds:
                #check consistency
                assert ds.crs.to_epsg() == self.crs.to_epsg()
                
                #check intersection 
                rbnds = sgeo.box(*ds.bounds)
                ebnds = sgeo.box(*gdf.total_bounds)
                if not bbox is None: #with the bounding box

                    assert bbox.within(ebnds), 'bounding box exceeds assets extent'                    
                    
                    #build a clean window
                    """basically rounding the raster window so everything fits"""
                    window, win_transform = get_window(ds, bbox)
 
                    
                    """
                    plt.close('all')
                    fig, ax = plt.subplots()
                    ax.plot(*rbnds.exterior.xy, color='red', label='raster (raw)')
                    ax.plot(*ebnds.exterior.xy, color='blue', label='assets')
                    gdf.plot(ax=ax, color='blue')
                    ax.plot(*bbox.exterior.xy, color='orange', label='bbox', linestyle='dashed')
                    wbnds = sgeo.box(*rio.windows.bounds(window, transform=ds.transform))
                    ax.plot(*wbnds.exterior.xy, color='green', label='window', linestyle='dotted')
                    #ax.plot(*bbox1.exterior.xy, color='black', label='bbox_buff', linestyle='dashed')
                    fig.legend()
                    limits = ax.axis()
                    """
                    
                    
                else:  #between assets and raster
                    assert ebnds.intersects(rbnds), 'raster and assets do not intersect'
                    
                    window=None
                    
 
                #===============================================================
                # loop each category's mask'
                #=============================================================== 
                mosaic_ar = load_array(ds, window=window) 
                
                cm_d = self.mosaic_to_masks(mosaic_ar)
                #load and compute zonal stats
                zd = dict()
                for catid, ar_raw in cm_d.items():
                    if np.any(ar_raw):
                        stats_d = zonal_stats(gdf, 
                                               np.where(ar_raw, 1, np.nan), 
                                                affine=win_transform, 
                                                nodata=0, 
                                                all_touched=False, #only centroids
                                                stats=[ 'count',
                                                       #'nan', #only interested in real interseects
                                                       ])
                        zd[catid] = pd.DataFrame(stats_d)
                        
 
                        """
                        plot_rast(ar_raw, transform=win_transform, ax=ax )
                        """
                #===============================================================
                # wrap
                #===============================================================
                res_d[scale] = pd.concat(zd, axis=1, names=['dsc']).droplevel(1, axis=1).rename_axis(gdf.index.name)
        
        #=======================================================================
        # merge
        #=======================================================================
        """dropping spatial data"""
        rdx = pd.concat(res_d, axis=1, names=['scale'])
        log.info('finished w/ %s' % str(rdx.shape))
        return rdx

class ExpoSession(ExpoWrkr, Agg2Session):
    """tools for experimenting with downsample sets"""
    
    def __init__(self, 
                 method='direct',
                 **kwargs):
        """
        
        Parameters
        ----------
 
        """
        
        super().__init__(obj_name='expo', scen_name=method,subdir=False, **kwargs)
        
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
        start=now()
        #=======================================================================
        # classification masks
        #=======================================================================
        df_raw = pd.read_pickle(pick_fp).loc[:, ['downscale', 'catMosaic']]
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
        gdf = gpd.read_file(finv_fp, bbox=bbox).rename_axis('fid')
        assert len(gdf)>0
        abnds = sgeo.box(*gdf.total_bounds) #bouds
        
        if not bbox is None:
            if not abnds.within(bbox):
                """can happen when an asset intersects the bbox"""
                log.warning('asset bounds  not within bounding box \n    %s\n    %s'%(
                            abnds.bounds, bbox.bounds))
        
        log.info('loaded %i feats (w/ aoi: %s) from \n    %s'%(
            len(gdf), type(bbox).__name__, os.path.basename(finv_fp)))
        
        assert gdf.crs==self.crs, 'crs mismatch'
        assert len(gdf)>0
        
        """
        
        tuple(np.round(gdf.total_bounds, 1).tolist())
        
        """
        #=======================================================================
        # get downscales
        #=======================================================================
        res_dx = self.get_assetRsc(cm_fp_d, gdf, 
                                   bbox=abnds, #using asset bounds because this might be larger than the bbox 
                                   logger=log)
        
        #=======================================================================
        # write
        #=======================================================================
        res_dx.to_pickle(ofp)
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(res_dx.shape), ofp))
        
        
        return ofp
 
                
 
        
        
        """
        gdf.plot()
        plt.show()
        """
        


        
        
        
        
        
    
        
 