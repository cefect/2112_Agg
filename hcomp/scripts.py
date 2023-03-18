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

from hp.oop import Session, Basic
from hp.rio import write_resample
from hp.hyd import get_wse_rlay, get_wsh_rlay
#from agg2.haz.scripts import UpsampleChild


class AggWSE(object):
    """simplified aggregation methods"""
    
    def get_avgWSH(self,
                   wsh1_fp, dem1_fp,
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
        
        
        
        
        
    def get_agg_byType(self, method, **kwargs):
        
        if method in ['direct', 'avgWSH']:
            func = self.get_avgWSH
        elif method in ['filter', 'avgWSE']:
            func = self.get_avgWSE
        else:
            raise KeyError(method)
        
        


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
        

        
    