'''
Created on Aug. 27, 2022

@author: cefect
'''
import numpy as np
import os, copy, datetime
import rasterio as rio
from definitions import wrk_dir
from hp.np import apply_blockwise, upsample 
from hp.oop import Session
from hp.rio import RioWrkr, assert_extent_equal, is_divisible, assert_rlay_simple, load_array

class Dsamp(RioWrkr, Session):
    """tools for experimenting with downsample sets"""
    
 
    
    def __init__(self, 
 
                 **kwargs):
        """
        
        Parameters
        ----------
        downscale: int, default 2
            multipler for new pixel resolution
            oldDimension*(1/downscale) = newDimension
 
        """
 
        
        super().__init__(obj_name='dsmp', wrk_dir=wrk_dir, **kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
 
        
    #===========================================================================
    # MAIN RUNNERS-----
    #===========================================================================
    def run_all(self,dem_fp, wse_fp,
 
                 dsc_l=None,
                 base_resolution=None,
                 downSampleIter_kwargs = dict(
                        reso_iters=5,
                        ),
                  
                 
                 method='direct',
                 **kwargs):
        """build downsample set
        
        Parameters
        -----------
        dem_fp: str
            filepath to DEM raster
        wse_fp: str
            filepath to WSE raster
        
        dsc_l: list, optional
            set of new resolutions to construct
            default: build from 

 
        """
        #=======================================================================
        # defaults
        #=======================================================================
 
        start = datetime.datetime.now()
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dsmp',  **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=tmp_dir, write=write)
        
        #=======================================================================
        # populate downsample set
        #=======================================================================
        if dsc_l is None:
            if base_resolution is None:
                with rio.open(dem_fp, mode='r') as dem_ds: 
                    dem_shape = dem_ds.shape
 
            dsc_l = self.get_downSampleIter(base_resolution=base_resolution, **downSampleIter_kwargs, **skwargs)
            
        #=======================================================================
        # build the set from this
        #=======================================================================
        self.build_dset(dem_fp, wse_fp, dsc_l=dsc_l, method=method)
            
    def get_downSampleIter(self,
                           base_resolution=None,
                           reso_iters=3,
 
                           **kwargs):
        """get a fibonaci like sequence around the base
        
        Parameters
        ---------

        base_resolution: int, optional
            base resolution from which to construct the dsc_l
            default: build from dem_fp
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dsmp',  **kwargs)
        
        assert isinstance(base_resolution, int)
        
        l = [base_resolution]
        for i in range(reso_iters-1):
            l.append(l[i]*2)
        return l
            
    def build_dset(self,
            dem_fp, wse_fp,
            dsc_l=None,
            method='direct',
            **kwargs):
        """build a set of downsampled rasters
        
        Parameters
        ------------
        method: str default 'direct'
            downsample routine method
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dsmp',  **kwargs)
        
        raise IOError('stopped here')
        
        return fp_d
        
        
    
    def _func_setup(self, dkey, 
                    logger=None, out_dir=None, tmp_dir=None,ofp=None,
 
                    write=None,layname=None,ext='.tif',
                    ):
        """common function default setup"""
        #logger
        if logger is None:
            logger = self.logger
        log = logger.getChild('build_%s'%dkey)
 
        
        #temporary directory
        if tmp_dir is None:
            tmp_dir = os.path.join(self.tmp_dir, dkey)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            
        #=======================================================================
        # #main outputs
        #=======================================================================
        if out_dir is None: out_dir = self.out_dir
        if write is None: write=self.write
        
        if layname is None:layname = '%s_%s'%(self.fancy_name, dkey)
         
        if ofp is None:
            if write:            
                ofp = os.path.join(out_dir, layname+ext)            
            else:
                ofp=os.path.join(tmp_dir, layname+ext)
            
        if os.path.exists(ofp):
            assert self.overwrite
            os.remove(ofp)
 
            
        return log, tmp_dir, out_dir, ofp, layname, write