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
    write_resample, assert_spatial_equal, assert_extent_equal, get_meta, load_array,
    write_array2, get_profile, rlay_ar_apply
    )
from hp.hyd import (
    get_wse_rlay, get_wsh_rlay, assert_dem_ar, assert_wse_ar, assert_wsh_ar,
    )
#from agg2.haz.scripts import UpsampleChild


class FloodGridAggregation(object):
    """simplified aggregation methods
    See also agg2.haz.scripts.UpsampleChild
    
    """
    
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
        log.info(f'building the WSE2')
        wse2_fp = get_wse_rlay(dem2_fp, wsh2_fp, ofp=os.path.join(out_dir, f'{resname}_WSE.tif'))
        
        return dem2_fp, wse2_fp, wsh2_fp
    
    def get_avgWSE(self,
                   dem1_fp, wse1_fp, 
                   aggscale=2,
                   resampling=Resampling.average,
                   **kwargs):
        """
        get aggregated flood grids from WSE averaging. formerly 'filter'
    
        Parameters
        ----------
        dem1_fp : str
            Filepath to the input DEM raster.
        wse1_fp : str
            Filepath to the input WSE raster.
        aggscale : int, optional
            The aggregation scale factor. Default is 2.
        resampling : Resampling method, optional
            The resampling method used when aggregating the rasters. Default is `Resampling.average`.
 
    
        Returns
        -------
        wse2_fp2 : str
            Filepath to the output WSE raster.
    
        Notes
        -----
        This function first aggregates the input DEM and WSE rasters using the specified aggregation scale factor and resampling method. Then it filters out invalid values in the WSE raster where it is higher than or equal to the DEM. Finally, it writes out a new WSE raster with filtered values.
    
        """
        
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
        wse2_fp1 = agg(wse1_fp, 'WSE')
        
        #=======================================================================
        # filter WSE
        #=======================================================================
        dem2_ar = load_array(dem2_fp, masked=True)
        assert_dem_ar(dem2_ar, msg='agg DEM')
        
        wse2_ar1 = load_array(wse2_fp1, masked=True)
        assert_wse_ar(wse2_ar1)
        
        #combine the masks
        bx = np.logical_or(
            dem2_ar.data>=wse2_ar1.data,
            wse2_ar1.mask)
        
        #rebuild
        wse2_ar2 = ma.array(wse2_ar1.data,mask=bx,fill_value=wse2_ar1.fill_value)
        
        #write the raster
        wse2_fp2 = write_array2(wse2_ar2, 
                     ofp=os.path.join(out_dir, f'{resname}_WSE2.tif'),
                     **get_profile(wse2_fp1))
        
        log.info(f'raw mask={wse2_ar1.mask.sum()} new mask={bx.sum()} wrote new WSE2 to \n    {wse2_fp2}')
        #=======================================================================
        # build WSH
        #=======================================================================
        log.info(f'building WSH2')
        wsh2_fp = get_wsh_rlay(dem2_fp, wse2_fp2,  ofp=os.path.join(out_dir, f'{resname}_WSH2.tif'))
        
        return dem2_fp, wse2_fp2, wsh2_fp
        
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
        # precheck
        #=======================================================================
        rlay_ar_apply(dem_fp, assert_dem_ar)
        #=======================================================================
        # prep WSH averaging
        #=======================================================================
        if method in ['direct', 'avgWSH']:
            #get missing input
            if wsh_fp is None:
                wsh_fp = get_wsh_rlay(dem_fp, wse_fp, out_dir=tmp_dir)
            
            #check
            assert_spatial_equal(dem_fp, wsh_fp, msg='DEM vs. WSH')
            rlay_ar_apply(wsh_fp, assert_wsh_ar) 
            
            #build
            func = lambda:self.get_avgWSH(dem_fp, wsh_fp,  **skwargs, **method_kwargs)
                
        #=======================================================================
        # prep WSE averaging
        #=======================================================================
        elif method in ['filter', 'avgWSE']:
            #get missing input
            if wse_fp is None:
                wse_fp = get_wse_rlay(dem_fp, wsh_fp, out_dir=tmp_dir)
                
            #check
            assert_spatial_equal(dem_fp, wse_fp, msg='DEM vs. WSE')
            rlay_ar_apply(wse_fp, assert_wse_ar)   
            
            #build
            func = lambda :self.get_avgWSE(dem_fp, wse_fp, **skwargs, **method_kwargs)
 
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
        
        


class HydCompareSession(Session, FloodGridAggregation):
    """session for comparing a single hyd pair vs. agg pair"""
    def __init__(self,**kwargs):
        super().__init__(obj_name='hcomp',  **kwargs)
        
 

        
    
