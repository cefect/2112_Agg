'''
Created on Mar. 18, 2023

@author: cefect
'''
import numpy as np
import numpy.ma as ma
import pandas as pd
import os, copy, datetime, gc
import rasterio as rio
from rasterio.enums import Resampling

from hp.basic import now

from hp.oop import Session, Basic
from hp.rio import (
    write_resample, assert_spatial_equal, assert_extent_equal, get_meta
    )
from hp.hyd import get_wse_rlay, get_wsh_rlay
#from agg2.haz.scripts import UpsampleChild


class AggWSE(object):
    """simplified aggregation methods"""
    
    def get_avgWSH(self,
                   dem1_fp, wsh1_fp, 
                   aggscale=2,
                   resampling=Resampling.average,
                   **kwargs):
        
        """direct aggregation of DEM and WD. WSE is recomputed. formerly 'direct'
        
        Pars
        -----------
        
        """
        log, tmp_dir, out_dir, _, resname, write = self._func_setup('avgWSH',  **kwargs)
        
        #=======================================================================
        # upscale/aggregate
        #=======================================================================
        def agg(fp, layerName):
 
            log.info(f'aggregating the {layerName} {os.path.basename(fp)} to aggscale={aggscale}')
            return write_resample(fp, resampling=resampling, scale=1/aggscale, 
                           ofp=os.path.join(out_dir, f'{resname}_{layerName}.tif'))
            
        #DEM
        dem2_fp = agg(dem1_fp, 'DEM')
        wsh2_fp = agg(wsh1_fp, 'WSH')
        
        #=======================================================================
        # build WSE
        #=======================================================================
        log.info(f'building the WSE')
        wse2_fp = get_wse_rlay(dem2_fp, wsh2_fp, ofp=os.path.join(out_dir, f'{resname}_WSE.tif'))
        
        return dem2_fp, wse2_fp, wsh2_fp
    
    def get_avgWSE(self,
                   dem1_fp, wse1_fp, 
                   aggscale=2,
                   **kwargs):
        
        log, tmp_dir, out_dir, _, resname, write = self._func_setup('avgWSE',  **kwargs)
        
        #=======================================================================
        # upscale/aggregate
        #=======================================================================
        def agg(fp, layerName):
 
            log.info(f'aggregating the {layerName} {os.path.basename(fp)} to aggscale={aggscale}')
            return write_resample(fp, resampling=resampling, scale=1/aggscale, 
                           ofp=os.path.join(out_dir, f'{resname}_{layerName}.tif'))
            
        #DEM
        dem2_fp = agg(dem1_fp, 'DEM')
        wsh2_fp = agg(wsh1_fp, 'WSH')
    
    
 
        
    def get_agg_byType(self, method, 
                       dem_fp=None,
                       wse_fp=None,
                       wsh_fp=None,
                       aggscale=2,
                       method_kwargs=dict(),
                       **kwargs):
        """
        Aggregates flood grids using the specified method.
    
        Parameters
        ----------
        method : str
            The aggregation method to use. Options include 'direct', 'avgWSH', 'filter', and 'avgWSE'.
        dem_fp : str, optional
            File path to the DEM raster layer. Default is None.
        wse_fp : str, optional
            File path to the WSE raster layer. Default is None.
        wsh_fp : str, optional
            File path to the WSH raster layer. Default is None.
        aggscale : int, optional
            The scale factor for aggregation. Default is 2.
        method_kwargs : dict, optional
            Additional keyword arguments for the specified aggregation method. Default is an empty dictionary.
        **kwargs :
            Additional keyword arguments.
    
        Returns
        -------
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, resname, write = self._func_setup('agg',  **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=out_dir, aggscale=aggscale)
        start = now()
        rlay_meta = get_meta(dem_fp, att_l=['height', 'width'])
        
 
        log.info(f'aggscale={aggscale} on \n    {rlay_meta}')
        #=======================================================================
        # prep WSH averaging
        #=======================================================================
        if method in ['direct', 'avgWSH']:
            #get missing input
            if wsh_fp is None:
                wsh_fp = get_wsh_rlay(dem_fp, wse_fp, out_dir=tmp_dir)
            
            assert_spatial_equal(dem_fp, wsh_fp, msg='DEM vs. WSH')    
            
            func = lambda:self.get_avgWSH(dem_fp, wsh_fp,  **skwargs, **method_kwargs)
                
        #=======================================================================
        # prep WSE averaging
        #=======================================================================
        elif method in ['filter', 'avgWSE']:
            #get missing input
            if wse_fp is None:
                wse_fp = get_wse_rlay(dem_fp, wsh_fp, out_dir=tmp_dir)
                
            assert_spatial_equal(dem_fp, wse_fp, msg='DEM vs. WSE')   
            
            func = lambda :self.get_avgWSE(wse_fp, dem_fp,  **skwargs, **method_kwargs)
 
        else:
            raise KeyError(method)
        
        #=======================================================================
        # execute
        #=======================================================================
        log.info(f'executing for {method}')
        try:
            dem2_fp, wse2_fp, wsh2_fp = func()
        except Exception as e:
            raise IOError(f'failed to get agg with method={method}\n    {e}')
        
        #=======================================================================
        # check
        #=======================================================================
        assert_extent_equal(dem_fp, dem2_fp, msg='DEM1 vs. DEM2')
        assert_spatial_equal(dem2_fp, wsh2_fp, msg='DEM2 vs. WSH2')
        assert_spatial_equal(dem2_fp, wse2_fp, msg='DEM2 vs. WSE2')
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = now()-start
        log.info(f'finished {method} in {tdelta.total_seconds()}')
        
        return dem2_fp, wse2_fp, wsh2_fp
        
        


class HydCompareSession(Session, AggWSE):
    """session for comparing a single hyd pair vs. agg pair"""
    def __init__(self,**kwargs):
        super().__init__(obj_name='hcomp',  **kwargs)
        
    
    def get_aggWSE(self,
                   method='direct', resampleAlg='average',
                   **kwargs):
        """build an aggreagated WSE using a named method"""
        
        #=======================================================================
        # setup
        #=======================================================================
        log, tmp_dir, _, ofp, resname, write = self._func_setup('dsmp',  **kwargs)
        
        #=======================================================================
        # execute
        #=======================================================================
        

        
    