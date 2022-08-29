'''
Created on Aug. 19, 2022

@author: cefect

classifying downsample type

let's try w/ mostly rasterio?
simple session (no retrieve)
'''
 

import numpy as np
import os, copy, datetime
import rasterio as rio
from definitions import wrk_dir 
from hp.np import apply_blockwise, upsample 
from hp.oop import Session
from hp.rio import RioWrkr, assert_extent_equal, is_divisible, assert_rlay_simple, load_array


 
class DsampClassifier(RioWrkr): 
    """shareable tools for build downsample classification masks"""
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
 
        
        super().__init__( **kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
        self.downscale=downscale
 

        
    
    #===========================================================================
    # UNIT BUILDRES-------
    #===========================================================================
    def build_crop(self, raw_fp, new_shape=None, divisor=None, **kwargs):
        """build a cropped raster which achieves even division
        
        Parameters
        ----------
        bounds : optional
        
        divisor: int, optional
            for computing the nearest whole division
        """
        #=======================================================================
        # defaults
        #=======================================================================
        rawName = os.path.basename(raw_fp).replace('.tif', '')[:6]
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('crops%s'%rawName,  **kwargs)
        
        assert isinstance(divisor, int)
        
        with RioWrkr(rlay_ref_fp=raw_fp, session=self) as wrkr:
            
            raw_ds = wrkr._base()
            
            """
            raw_ds.read(1)
            """
            #=======================================================================
            # precheck
            #=======================================================================
            assert not is_divisible(raw_ds, divisor), 'no need to crop'            
 
            #===================================================================
            # compute new_shape
            #===================================================================
            if new_shape is None: 
                new_shape = tuple([(d//divisor)*divisor for d in raw_ds.shape])
                
            log.info('cropping %s to %s for even divison by %i'%(
                raw_ds.shape, new_shape, divisor))
                
 
            
            self.crop(rio.windows.Window(0,0, new_shape[1], new_shape[0]), dataset=raw_ds,
                      ofp=ofp, logger=log)
            
        return ofp
            
 
                

 
    
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
        rawName = os.path.basename(raw_fp).replace('.tif', '')[:6]
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('coarse%s'%rawName,  **kwargs)
        
        if downscale is None: downscale=self.downscale
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(downscale, int)
        assert downscale>1
        
        if __debug__:
            #===================================================================
            # check we have a divisible shape
            #===================================================================
            with rio.open(raw_fp, mode='r') as dem_ds:
 
                dem_shape = dem_ds.shape
                
                assert_rlay_simple(dem_ds, msg='DEM')
 
                
            #check shape divisibility
            for dim in dem_shape:
                assert dim%downscale==0, 'unequal dimension (%i/%i -> %.2f)'%(dim, downscale, dim%downscale)
                
 
            
        #=======================================================================
        # downsample
        #=======================================================================
        resampling = getattr(rio.enums.Resampling, resampleAlg)
        with RioWrkr(rlay_ref_fp=raw_fp, session=self) as wrkr:
            
            self._check_dem_ar(wrkr._base().read(1)) 
            
            res_ds = wrkr.resample(resampling=resampling, scale=1/downscale)
            wrkr.write_memDataset(res_ds, ofp=ofp)
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        #mstore.removeAllMapLayers()
        assert os.path.exists(ofp)
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
            #===================================================================
            # wse_ds = wrkr.open_dataset(wse_fp)
            # wse_ar = wse_ds.read(1)
            #===================================================================
            wse_ar = load_array(wse_fp)
            
            assert dem_ar.shape==wse_ar.shape
 
            
            #===================================================================
            # compute
            #===================================================================
            delta_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
            
            assert np.all(delta_ar>=0)
            
            wrkr.write_array(delta_ar, ofp=ofp, logger=log)
            
        return ofp
    
    
    




    
    
    def get_catMasks(self,
                     downscale=None,
                     dem_ds=None,
                     wse_ds=None,
 
                     **kwargs):
        """compute the a mask for each downsample category
        
        
        """
        
        #=======================================================================
        # defautls
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('cm',  **kwargs)
        if downscale is None: downscale=self.downscale
        
        assert isinstance(downscale, int)
        
        #===================================================================
        # load layers----
        #===================================================================
        #fine dem
 
        dem_ar = load_array(dem_ds)
        
        self._check_dem_ar(dem_ar)
 
        assert_extent_equal(dem_ds, wse_ds)        
        
        wse_ar = load_array(wse_ds)
            
        assert dem_ar.shape==wse_ar.shape
        assert np.isnan(wse_ar).any(), 'wse should have null where dry'
        if not np.all(dem_ar>0): log.warning('got some negative terrain values!')
        assert np.all(wse_ar[~np.isnan(wse_ar)]>0)
        
        #=======================================================================
        # globals
        #=======================================================================
        def apply_upsample(ar, func):
            arC = apply_blockwise(ar, func, n=downscale) #max of each coarse block
            return upsample(arC,n=downscale) #scale back up
        
        def log_status(k):
            log.info('    calcd %i/%i \'%s\''%(res_d[k].sum(), dem_ar.size, k))
            
        res_d = dict()
        #===================================================================
        # compute delta
        #===================================================================
        log.info('computing deltas')
        delta_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
        assert np.all(delta_ar>=0)
        
        #=======================================================================
        # #dry-dry: max(delta) <=0
        #=======================================================================
        delta_max_ar = apply_upsample(delta_ar, np.max)
        
        res_d['DD'] = delta_max_ar<=0
        log_status('DD')
        #===================================================================
        # #wet-wet: min(delta) >0
        #===================================================================
        delta_min_ar = apply_upsample(delta_ar, np.min)
        
        res_d['WW'] = delta_min_ar>0
        log_status('WW')
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
            log.info('    flagged %i/%i partials'%(partial_bool_ar.sum(), partial_bool_ar.size))
    

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
        log_status('WP')
        
        #dry-partials: mean(DEM)>mean(WSE)
        res_d['DP'] = np.logical_and(partial_bool_ar,
                                     dem_mean_ar>=wse_mean_ar)
        
        log_status('DP')
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
        
        return res_d
        

    def get_catMosaic(self, cm_d, cm_int_d=None, logger=None):
        """moasic together the four dsc cat masks
        
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
        if logger is None: logger=self.logger
        log=logger.getChild('get_cMos')
        
        if cm_int_d is None:
            cm_int_d = self.cm_int_d.copy()
        #=======================================================================
        # precheck
        #=======================================================================
        miss_l = set(cm_int_d.keys()).symmetric_difference(cm_d.keys())
        assert len(miss_l) == 0, miss_l
        assert np.all(np.add.reduce(list(cm_d.values())) == 1), 'discontinuous masks'
        #=======================================================================
        # loop and build
        #=======================================================================
        res_ar = np.full(cm_d['DD'].shape, np.nan, dtype=np.int32) #build an empty dumm
        for k, mar in cm_d.items():
            log.info('for %s setting %i' % (k, mar.sum()))
            np.place(res_ar, mar, cm_int_d[k])
        
        #=======================================================================
        # check
        #=======================================================================
        assert np.all(res_ar % 2 == 1), 'failed to get all odd values'
        return res_ar
            

    #===========================================================================
    # PRIVATE HELPERS---------
    #===========================================================================
    def _check_dem_ar(self, ar):
        """check dem array satisfies assumptions"""
        #assert np.all(ar>0) #relaxing this
        assert np.all(~np.isnan(ar))
        assert 'float' in ar.dtype.name
        
        
 
    
    
class DsampClassifierSession(DsampClassifier, Session):
    """standalone session for downsample classification"""
 
    
    def __init__(self, 
 
                 obj_name='dsc',
 
                 **kwargs):
 
 
        
        super().__init__(obj_name=obj_name, wrk_dir=wrk_dir,  **kwargs)
        
    #===========================================================================
    # MAIN RUNNERS-----
    #===========================================================================
    def run_all(self,demR_fp, wseR_fp,
                demC_fp=None,
                 downscale=None, **kwargs):
        """prep layers and build downsample classification 
        
        
        Parameters
        ----------
        demC_fp: str optional
            filepath to the coarse DEM. Otherwise, this is built from teh raw/fine DEM
 
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if downscale is None: downscale=self.downscale
        start = datetime.datetime.now()
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dsc_%03i'%downscale,  **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=tmp_dir, write=write)
        
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert_extent_equal(demR_fp, wseR_fp)
        #=======================================================================
        # check divisibility
        #=======================================================================
        if not is_divisible(demR_fp, downscale):
            log.warning('uneven division w/ %i... clipping'%downscale)
            
            dem_fp = self.build_crop(demR_fp, divisor=downscale, **skwargs)
            wse_fp = self.build_crop(wseR_fp, divisor=downscale, **skwargs)
            
        else:
            dem_fp, wse_fp = demR_fp, wseR_fp
 
        
        
        #=======================================================================
        # algo
        #=======================================================================
        #build coarse dem
        #=======================================================================
        # if demC_fp is None:
        #     demC_fp = self.build_coarse(dem_fp, downscale=downscale, **skwargs)
        #=======================================================================
            
        #each mask
        cm_d, _ = self.build_cat_masks(dem_fp, wse_fp, **skwargs)
        
        #moasic together
        cm_fp = self.build_cat_mosaic(cm_d,ofp=ofp, **skwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished in %s'%(datetime.datetime.now()-start))
        return ofp
    
    def build_cat_masks(self,
                        dem_fp, wse_fp, 
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
 
        #=======================================================================
        # exec
        #=======================================================================
        ofp_d=dict()
        with DsampClassifier(rlay_ref_fp=dem_fp, session=self, downscale=downscale) as wrkr:
            #load the layers
            wse_ds = wrkr.open_dataset(wse_fp)
            dem_ds = wrkr._base()
            
            #build the masks
            res_d = wrkr.get_catMasks(dem_ds=dem_ds, wse_ds=wse_ds)
            
            
            #write masks
            if write:
                for k, mar in res_d.items():
                    ofp_d[k] = wrkr.write_array(mar.astype(int), ofp=os.path.join(out_dir, '%s_mask.tif'%k))
            
            
            
        log.info('finished writing %i'%len(ofp_d))
        return res_d, ofp_d
    
    def build_cat_mosaic(self, cm_d,
                         cm_int_d=None,
                         output_kwargs={},
                         **kwargs):
        """
        construct a mosaic from the 4 category masks
        

        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('cmMosaic',  **kwargs)
        res_ar = self.get_catMosaic(cm_d, cm_int_d=cm_int_d)
        
 
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.write_array(res_ar, ofp=ofp, logger=log, **output_kwargs)
        
        return ofp
    

def runr(
        dem_fp=None, wse_fp=None,
        **kwargs):
    with DsampClassifier(rlay_ref_fp=dem_fp, **kwargs) as ses:
        ofp = ses.run_all(dem_fp, wse_fp)
        
    return ofp
 
 
    
