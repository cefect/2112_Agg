'''
Created on Aug. 19, 2022

@author: cefect

classifying downsample type

let's try w/ mostly rasterio?
simple session (no retrieve)
'''
import os
from qgis.core import QgsMapLayerStore
import rasterio as rio
import numpy as np



from hp.Q import assert_rlay_equal, assert_extent_equal, RasterCalc, view, vlay_get_fdata, Qproj, assert_rlay_simple
from hp.oop import Session
from hp.basic import get_dict_str
from hp.rio import RioWrkr
from hp.np import apply_blockwise, upsample

class DsampClassifier(Qproj, Session):
    """tools for build downsample classification masks"""
    def __init__(self,
 
                 #requireed input files
 
                 
                 zthresh=0.001,
                 
 
                 **kwargs):
        """
        
        Parameters
        ----------
 
            filepath to digital elevation model raster
            
        zthresh: float, default 0.001
            zero threshold
        """
 
        
        super().__init__(**kwargs)
 
        
        #=======================================================================
        # attach
        #=======================================================================
        self.zthresh=zthresh
        
 
 

    def build_coarse(self,
                        raw_fp,
                        downscale=2,
                        resampleAlg='average',
                        **kwargs):
        
        """
        construct a coarse raster from a raw raster
        
        Parameters
        ----------
        raw_fp: str
            filepath to fine raster
        
        downscale: int, default 2
            multipler for new pixel resolution
            oldDimension*(1/downscale) = newDimension
        """
        rawName = os.path.basename(raw_fp).replace('.tif', '')
        
        log, mstore, tmp_dir, out_dir, ofp, layname = self._func_setup('coarse%s'%rawName,  **kwargs)
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(downscale, int)
        assert downscale>1
        
        if __debug__:
            rlay_raw = self.rlay_load(raw_fp, mstore=mstore, logger=log)
            stats_d = self.rlay_get_stats(rlay_raw, logger=log)
            assert stats_d['MIN']>0
            if stats_d['noData_cnt']>0:
                log.warning('got %i/%i nodata cells on %s'%(
                    stats_d['noData_cnt'], stats_d['cell_cnt'], rawName))
                
            #check we have a clean division
            for dim in ['height', 'width']:
                
                #check we have enough fine cells to make at least 1 new aggregated cell
                assert stats_d[dim]>=downscale, 'insufficient cells for specified aggregation'
                
                if not stats_d[dim]%downscale==0:
                    log.warning('uneven division for \'%s\' of %i/%i (%.2f)'%(
                        dim, stats_d[dim], downscale, stats_d[dim]%downscale))
            
            assert_rlay_simple(rlay_raw)
            
        #=======================================================================
        # downsample
        #=======================================================================
        resampling = getattr(rio.enums.Resampling, resampleAlg)
        with RioWrkr(rlay_ref_fp=raw_fp, session=self) as wrkr:
            wrkr.resample(resampling=resampling, scale=1/downscale, ofp=ofp)
            
        #=======================================================================
        # wrap
        #=======================================================================
        mstore.removeAllMapLayers()
        
        return ofp
    
    def build_delta(self, 
                    dem_fp, wse_fp,
                    **kwargs):
        
        """build DEM WSE delta
        
        this is a bit too simple...
        """
        
        log, mstore, tmp_dir, out_dir, ofp, layname = self._func_setup('delta',  **kwargs)
        
        
        with RioWrkr(rlay_ref_fp=dem_fp, session=self) as wrkr:
            
            #===================================================================
            # load layers----
            #===================================================================
            #dem
            dem_ar = wrkr._base().read(1)
            
            #load the wse
            wse_ds = wrkr.open_dataset(wse_fp)
            wse_ar = wse_ds.read(1)
            
            assert dem_ar.shape==wse_ar.shape
 
            
            #===================================================================
            # compute
            #===================================================================
            delta_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
            
            assert np.all(delta_ar>=0)
            
            wrkr.write_dataset(delta_ar, ofp=ofp, logger=log)
            
        return ofp
    
    def build_cat_masks(self,
                        dem_fp, demC_fp, wse_fp, 
                        downscale=2,
                        write=None,
                        **kwargs):
        
        """build masks for each category
        
        Parameters
        -----------
        dem_fp: str
            filepath to fine/raw DEM
        demC_fp: str
            filepath to coarse/downsampled DEM
        wse_fp: str
            filepath  to fine WSE dem
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, mstore, tmp_dir, out_dir, ofp, layname = self._func_setup('cmask',  **kwargs)
        if write is None: write=self.write
        
        res_d = dict()
        
        def apply_upsample(ar, func):
            arC = apply_blockwise(ar, func, n=downscale) #max of each coarse block
            return upsample(arC,n=downscale) #scale back up
        
        with RioWrkr(rlay_ref_fp=dem_fp, session=self) as wrkr:
            
            #===================================================================
            # load layers----
            #===================================================================
            #fine dem
            dem_ar = wrkr._base().read(1)
            
            assert not np.isnan(dem_ar).any(), 'dem should have no nulls'
            
            #load the wse
            wse_ds = wrkr.open_dataset(wse_fp)
            wse_ar = wse_ds.read(1)
            
            assert dem_ar.shape==wse_ar.shape
            assert np.isnan(wse_ar).any(), 'wse should have null where dry'
            
            #===================================================================
            # compute delta
            #===================================================================
            delta_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
            assert np.all(delta_ar>=0)
            
            #=======================================================================
            # #dry-dry: max(delta) <=0
            #=======================================================================
            delta_max_ar = apply_upsample(delta_ar, np.max)
            
            res_d['DD'] = delta_max_ar<=0
            
            #===================================================================
            # #wet-wet: min(delta) >0
            #===================================================================
            delta_min_ar = apply_upsample(delta_ar, np.min)
            
            res_d['WW'] = delta_min_ar>0
            
            #===================================================================
            # #partials: max(delta)>0 AND min(delta)==0
            #===================================================================
            partial_bool_ar = np.logical_and(
                delta_max_ar>0,delta_min_ar==0)
            
            #check this is all remainers
            assert partial_bool_ar.sum() + res_d['WW'].sum() + res_d['DD'].sum() == partial_bool_ar.size
            
            if not np.any(partial_bool_ar):
                log.warning('no partials!')
            else:
                log.info('flagged %i/%i partials'%(partial_bool_ar.sum(), partial_bool_ar.size))
        

            #===============================================================
            # compute means
            #===============================================================
            dem_mean_ar = apply_upsample(dem_ar, np.mean)
            wse_mean_ar = apply_upsample(wse_ar, np.nanmean) #ignore nulls in denomenator
            #===============================================================
            # #wet-partials: mean(DEM)<mean(WSE)
            #===============================================================
            res_d['WP'] = np.logical_and(partial_bool_ar,
                                         dem_mean_ar<wse_mean_ar)
            
            #dry-partials: mean(DEM)>mean(WSE)
            res_d['DP'] = np.logical_and(partial_bool_ar,
                                         dem_mean_ar<=wse_mean_ar)
            
            #===================================================================
            # compute stats
            #===================================================================
            stats_d = {k:ar.sum()/ar.size for k, ar in res_d.items()}
            
            
            log.info('computed w/ \n    %s'%stats_d)
            
            #===================================================================
            # output rasteres
            #===================================================================
            
            ofp_d = dict()
            if write:
                for k, mar in res_d.items():
                    ofp_d[k] = wrkr.write_dataset(mar.astype(int), ofp=os.path.join(out_dir, '%s_mask.tif'%k))
            
        log.info('finished writing %i'%len(ofp_d))
        return res_d, ofp_d
    
    
    def build_cat_mosaic(self):
        raise IOError('stopped here')
            

    #===========================================================================
    # PRIVATE HELPERS---------
    #===========================================================================
    def _func_setup(self, dkey, 
                    logger=None, out_dir=None, 
                    mstore=None,
                    write=None,layname=None,ext='.tif',
                    ):
        """common function default setup"""
        #logger
        if logger is None:
            logger = self.logger
        log = logger.getChild('build_%s'%dkey)
 
        #QGIS
        if mstore is None:
            mstore = QgsMapLayerStore()
        
        #temporary directory
        tmp_dir = os.path.join(self.tmp_dir, dkey)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            
        #=======================================================================
        # #main outputs
        #=======================================================================
        if out_dir is None: out_dir = self.out_dir
        if write is None: write=self.write
        
        if layname is None:layname = '%s_%s'%(self.fancy_name, dkey)
         
        if write:            
            ofp = os.path.join(out_dir, layname+ext)            
        else:
            ofp=os.path.join(tmp_dir, layname+ext)
            
        if os.path.exists(ofp):
            assert self.overwrite
            os.remove(ofp)
 
            
        return log, mstore, tmp_dir, out_dir, ofp, layname
    
    
def get_wse_filtered(wse_raw_ar, dem_ar):
    """mask out negative WSE values"""
    wse_ar = wse_raw_ar.copy()
    np.place(wse_ar, wse_raw_ar<=dem_ar, np.nan)
    
    return wse_ar
    

def run():
    with DsampClassifier() as ses:
        pass
    
        #=======================================================================
        # prep layers
        #=======================================================================
        #load fine DEM
        
        #build coarse DEM
        
        #load fine WSE
        

        
        #build coarse WSE (no! dont need this)
        
        #=======================================================================
        # compute classes
        #=======================================================================
        #build fine delta. (useful for later)
            #check all above zero. 
            #fillna=0 
            
        #build a mask for each class
            
            #dry-dry: max(delta) <=0
            
            #wet-wet: min(delta) >0
            
            #partials: max(delta)>0 AND min(delta)==0
                #check this is all remainers
                
                #wet-partials: mean(DEM)<mean(WSE)
                
                #dry-partials: mean(DEM)>mean(WSE)
            
        #combine masks
            
            
    
        

if __name__ == "__main__": 
    run()
    
