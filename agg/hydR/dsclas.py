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

class DsampClassifier(RioWrkr, Qproj, Session):
    """tools for build downsample classification masks"""
    
    #integer maps for buildilng the mosaic
    cm_int_d = {'DD':11, 'WW':21, 'WP':31, 'DP':41}
    
    def __init__(self, 
                 downscale=2,
                 **kwargs):
        """
        
        Parameters
        ----------
        downscale: int, default 2
            multipler for new pixel resolution
            oldDimension*(1/downscale) = newDimension
 
        """
 
        
        super().__init__(**kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
        self.downscale=downscale
 
        
 
    #===========================================================================
    # MAIN RUNNERS-----
    #===========================================================================
    def run_all(self,dem_fp, wse_fp,
                demC_fp=None,
                 downscale=None, **kwargs):
        """prep all layers from fine/raw DEM and WSE
        
        
        Parameters
        ----------
        demC_fp: str optional
            filepath to the coarse DEM. Otherwise, this is built from teh raw/fine DEM
 
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('run',  **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=out_dir, write=write)
        if downscale is None: downscale=self.downscale
        #=======================================================================
        # algo
        #=======================================================================
        #build coarse dem
        if demC_fp is None:
            demC_fp = self.build_coarse(dem_fp, downscale=downscale, **skwargs)
            
        #each mask
        cm_d, _ = self.build_cat_masks(dem_fp, demC_fp, wse_fp, **skwargs)
        
        #moasic together
        cm_ar, _ = self.build_cat_mosaic(cm_d,ofp=ofp, **skwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return ofp
        
    
    #===========================================================================
    # UNIT BUILDRES-------
    #===========================================================================
    def build_coarse(self,
                        raw_fp,
                        downscale=None,
                        resampleAlg='average',
                        **kwargs):
        
        """
        construct a coarse raster from a raw raster
        
        Parameters
        ----------
        raw_fp: str
            filepath to fine raster
 
        """
        #=======================================================================
        # defaults
        #=======================================================================
        rawName = os.path.basename(raw_fp).replace('.tif', '')
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('coarse%s'%rawName,  **kwargs)
        
        if downscale is None: downscale=self.downscale
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(downscale, int)
        assert downscale>1
        
        #=======================================================================
        # if __debug__:
        #     rlay_raw = self.rlay_load(raw_fp, mstore=logger=log)
        #     stats_d = self.rlay_get_stats(rlay_raw, logger=log)
        #     assert stats_d['MIN']>0
        #     if stats_d['noData_cnt']>0:
        #         log.warning('got %i/%i nodata cells on %s'%(
        #             stats_d['noData_cnt'], stats_d['cell_cnt'], rawName))
        #         
        #     #check we have a clean division
        #     for dim in ['height', 'width']:
        #         
        #         #check we have enough fine cells to make at least 1 new aggregated cell
        #         assert stats_d[dim]>=downscale, 'insufficient cells for specified aggregation'
        #         
        #         if not stats_d[dim]%downscale==0:
        #             log.warning('uneven division for \'%s\' of %i/%i (%.2f)'%(
        #                 dim, stats_d[dim], downscale, stats_d[dim]%downscale))
        #     
        #     assert_rlay_simple(rlay_raw)
        #=======================================================================
            
        #=======================================================================
        # downsample
        #=======================================================================
        resampling = getattr(rio.enums.Resampling, resampleAlg)
        with RioWrkr(rlay_ref_fp=raw_fp, session=self) as wrkr:
            wrkr.resample(resampling=resampling, scale=1/downscale, ofp=ofp)
            
        #=======================================================================
        # wrap
        #=======================================================================
        #mstore.removeAllMapLayers()
        
        return ofp
    
    def build_delta(self, 
                    dem_fp, wse_fp,
                    **kwargs):
        
        """build DEM WSE delta
        
        this is a bit too simple...
        """
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('delta',  **kwargs)
        
        
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
                        downscale=None,
 
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
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('cmask',  **kwargs)
        
        if downscale is None: downscale=self.downscale
        
        res_d = dict()
        
        #=======================================================================
        # globals
        #=======================================================================
        def apply_upsample(ar, func):
            arC = apply_blockwise(ar, func, n=downscale) #max of each coarse block
            return upsample(arC,n=downscale) #scale back up
        
        #=======================================================================
        # exec
        #=======================================================================
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
                                         dem_mean_ar>=wse_mean_ar)
            
            #===================================================================
            # compute stats
            #===================================================================
            stats_d = {k:ar.sum()/ar.size for k, ar in res_d.items()}
            
            
            log.info('computed w/ \n    %s'%stats_d)
            
            #===================================================================
            # check
            #===================================================================
            chk_ar = np.add.reduce(list(res_d.values()))==1
            assert np.all(chk_ar), '%i/%i failed logic'%((~chk_ar).sum(), chk_ar.size)
            
            #===================================================================
            # output rasteres
            #===================================================================
            
            ofp_d = dict()
            if write:
                for k, mar in res_d.items():
                    ofp_d[k] = wrkr.write_dataset(mar.astype(int), ofp=os.path.join(out_dir, '%s_mask.tif'%k))
            
        log.info('finished writing %i'%len(ofp_d))
        return res_d, ofp_d
    
    def build_cat_mosaic(self, cm_d,
                         cm_int_d=None,
                         output_kwargs={},
                         **kwargs):
        """
        construct a mosaic from the 4 category masks
        
        Parameters
        ------------
        cm_d: dict
            four masks from build_cat_masks {category label: np.ndarray}
        
        cm_int_d: dict
            integer mappings for each category
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('cmMosaic',  **kwargs)
        if cm_int_d is None: cm_int_d=self.cm_int_d.copy()
        
        #=======================================================================
        # precheck
        #=======================================================================
        miss_l = set(cm_int_d.keys()).symmetric_difference(cm_d.keys())
        assert len(miss_l)==0, miss_l
        
        assert np.all(np.add.reduce(list(cm_d.values()))==1), 'discontinuous masks'
        #=======================================================================
        # loop and build
        #=======================================================================
        res_ar = np.full(cm_d['DD'].shape, np.nan, dtype=np.int32) #build an empty dumm
        
        for k, mar in cm_d.items():
            log.info('for %s setting %i'%(k, mar.sum())) 
            np.place(res_ar, mar, cm_int_d[k])
            
        #=======================================================================
        # check
        #=======================================================================
        assert np.all(res_ar%2==1), 'failed to get all odd values'
        
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.write_dataset(res_ar, ofp=ofp, logger=log, **output_kwargs)
        
        return res_ar, ofp
            

    #===========================================================================
    # PRIVATE HELPERS---------
    #===========================================================================
    def _func_setup(self, dkey, 
                    logger=None, out_dir=None, tmp_dir=None,ofp=None,
                    #mstore=None,
                    write=None,layname=None,ext='.tif',
                    ):
        """common function default setup"""
        #logger
        if logger is None:
            logger = self.logger
        log = logger.getChild('build_%s'%dkey)
 
        #QGIS
        #=======================================================================
        # if mstore is None:
        #     mstore = QgsMapLayerStore()
        #=======================================================================
        
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
    
