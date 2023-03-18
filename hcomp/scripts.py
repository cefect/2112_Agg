'''
Created on Mar. 18, 2023

@author: cefect
'''
import os, copy, datetime 
import numpy as np
import numpy.ma as ma
 

import rasterio as rio
import shapely.geometry as sgeo
from rasterio.enums import Resampling

from hp.basic import now

from hp.oop import Session 
from hp.rio import (
    write_resample, assert_spatial_equal, assert_extent_equal, get_meta, load_array,
    write_array2, get_profile, rlay_ar_apply, write_clip, get_rlay_shape
    )
from hp.hyd import (
    get_wse_rlay, get_wsh_rlay, assert_dem_ar, assert_wse_ar, assert_wsh_ar,
    )

from hp.fiona import get_bbox_and_crs
 

from hcomp.agg import FloodGridAggregation


class HydCompareSession(Session, FloodGridAggregation):
    """session for comparing a single hyd pair vs. agg pair"""
    def __init__(self,**kwargs):
        super().__init__(obj_name='hcomp',  **kwargs)
        
    def clip_rlays(self, raster_fp_d, aoi_fp=None, 
                   bbox=None, crs=None,
                 sfx='clip', **kwargs):
        """clip a dicrionary of raster filepaths
        
        Parameters
        -----------
        raster_fp_d: dict
            {key:filepath to raster}
            
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, resname, write = self._func_setup('clip_set',  **kwargs)
     
        #=======================================================================
        # retrive clipping parameters
        #=======================================================================
        if not aoi_fp is None:
            log.debug(f'clipping from aoi\n    {aoi_fp}')
            assert bbox is None
            assert crs is None        
            bbox, crs = get_bbox_and_crs(aoi_fp)
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(raster_fp_d, dict)        
        assert isinstance(bbox, sgeo.Polygon)
        assert hasattr(bbox, 'bounds')
        
        #=======================================================================
        # clip each
        #=======================================================================
        log.info(f'clipping {len(raster_fp_d)} rasters to \n    {bbox}')
        res_d = dict()
 
        for key, fp in raster_fp_d.items(): 
            d={'og_fp':fp}
            d['clip_fp'], d['stats'] = write_clip(fp,bbox=bbox,crs=crs,
                                                  ofp=os.path.join(out_dir, f'{key}_{sfx}.tif')
                                                  )
            
            log.debug(f'clipped {key}:\n    {fp}\n    %s'%d['clip_fp'])
            
            res_d[key] = d
            
        log.info(f'finished on {len(res_d)}')
        return res_d
    
    def get_agg_set(self,
                    method_lib,
                       dem_fp=None,
                       wse_fp=None,
                       wsh_fp=None,
                       aggscale=2,
                    **kwargs):
        """get aggregate grids for multiple methods"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, resname, write = self._func_setup('aggSet',  **kwargs)
        
        #=======================================================================
        # loop on each method
        #=======================================================================
        log.info(f'on {len(method_lib)} methods')
        agg_res_lib = dict()
        for method, method_kwargs in method_lib.items():
            d=dict()
            d['dem2'], d['wse2'], d['wsh2'] = self.get_agg_byType(method, 
                               dem_fp=dem_fp, wse_fp=wse_fp, wsh_fp=wsh_fp,
                               aggscale=aggscale, method_kwargs=method_kwargs)
            
            agg_res_lib[method]=d
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished')
        return agg_res_lib
                    
                    
                    
    
    def get_support_ratio(self,
                          fp_top, fp_bot,
                          ):
        """get scale difference"""
        shape1 = get_rlay_shape(fp_top)
        shape2 = get_rlay_shape(fp_bot)
        
        height_ratio = shape1[0]/shape2[0]
        width_ratio = shape1[1]/shape2[1]
        
        assert height_ratio==width_ratio, f'ratio mismatch. height={height_ratio}. width={width_ratio}'
        
        return width_ratio
        
        
        
        
        
        
                

 

        
    
