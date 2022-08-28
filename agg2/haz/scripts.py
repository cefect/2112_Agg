'''
Created on Aug. 27, 2022

@author: cefect
'''
import numpy as np
import pandas as pd
import os, copy, datetime
import rasterio as rio

from skimage.transform import downscale_local_mean
from definitions import wrk_dir
from hp.np import apply_blockwise, upsample 
from hp.oop import Session
from hp.rio import RioWrkr, assert_extent_equal, is_divisible, assert_rlay_simple, load_array, resample
from agg2.haz.dsc.scripts import DsampClassifier


def now():
    return datetime.datetime.now()

class DownsampleChild(RioWrkr):
    """child for performing a single downsample set
    
    NOTES
    -------
    I thought it cleaner, and more generalizeable, to keep this on a separate worker"""
    
 
    
    def __init__(self, 
                 downscale=1,
                 obj_name=None,
                 subdir=True,
                 **kwargs):
        """
        
        Parameters
        ---------
         downscale: int, default 1
            multipler for new pixel resolution
            oldDimension*(1/downscale) = newDimension
 
        """
        #=======================================================================
        # build defaults
        #=======================================================================
        if obj_name is None: obj_name='dsc%03i'%downscale
 
        super().__init__(subdir=subdir,obj_name=obj_name,
                          **kwargs)
        
        #=======================================================================
        # attachments
        #=======================================================================
        self.downscale=downscale
        
    def downscale_direct(self,
                         ds_d,
                         resampleAlg='average',
                         downscale=None,
                         **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, layname, write = self._func_setup('direct',  **kwargs)
        if downscale is None: downscale=self.downscale
        start = now()
        #=======================================================================
        # downscale each
        #=======================================================================
        log.info('downscale=%i on %s'%(downscale, list(ds_d.keys())))
        res_d = dict()
        for k, raw_ds in ds_d.items():
            ds1 = self.resample(dataset=raw_ds, resampling=getattr(rio.enums.Resampling, resampleAlg), scale=1/downscale)
            
            #write it
            res_d[k] = self.write_memDataset(ds1, dtype=np.float32,
                       ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),
                       logger=log)
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished downscaling and writing %i in %.2f secs'%(len(ds_d), (now()-start).total_seconds()))
        
        return res_d
            
    def downscale_filter(self,
                         ds_d,
                         resampleAlg='average',
                         downscale=None,
                         **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, layname, write = self._func_setup('filter',  **kwargs)
        if downscale is None: downscale=self.downscale
        start = now()
        """
        
        load_array(ds_d['wse'])
        """
 
        #=======================================================================
        # downscale dem and wse
        #=======================================================================
        log.info('downscale=%i on %s'%(downscale, list(ds_d.keys())))
 
        wse_ds1 = self.resample(dataset=ds_d['wse'], resampling=getattr(rio.enums.Resampling, resampleAlg), scale=1/downscale)

        wse_ar1 = load_array(wse_ds1).astype(np.float32)
        
        wse_ds1.close() #dont need this anymore
            
        dem_ds1 = self.resample(dataset=ds_d['dem'], resampling=getattr(rio.enums.Resampling, resampleAlg), scale=1/downscale)
        dem_ar1 = load_array(dem_ds1).astype(np.float32)
        
        self._base_inherit(ds=dem_ds1) #set class defaults from this
        #=======================================================================
        # filter wse
        #=======================================================================
        wse_ar2 = wse_ar1.copy()
        np.place(wse_ar2, wse_ar1<=dem_ar1, np.nan)
        wd_ds1 = self.load_memDataset(wse_ar2, name='wd')
        
        #=======================================================================
        # subtract to get depths
        #=======================================================================
        wd_ar = np.nan_to_num(wse_ar2-dem_ar1, nan=0.0).astype(np.float32)
        wd_ds = self.load_memDataset(wd_ar, name='wd')
        
        #=======================================================================
        # write all
        #=======================================================================
        res_d=dict()
        for k, ds in {'dem':dem_ds1, 'wse':wd_ds1, 'wd':wd_ds}.items():
            res_d[k] = self.write_memDataset(ds, dtype=np.float32,
                       ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),
                       logger=log)
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished downscaling and writing %i in %.2f secs'%(len(ds_d), (now()-start).total_seconds()))
        
        return res_d
        
        
    def write_dataset_d(self,
                        ar_d,
                        logger=None,
                        out_dir=None,
                         **kwargs):
        """helper for writing three main rasters consistently"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('write_dsd')
        if out_dir is None: out_dir=self.out_dir
        #=======================================================================
        # precheck
        #=======================================================================
        miss_l = set(['dem', 'wse', 'wd']).symmetric_difference(ar_d.keys())
        assert len(miss_l)==0
        
        log.info('writing %i to %s'%(len(ar_d), out_dir))
        
        """
        self.transform
        """
        
        #=======================================================================
        # loop and write
        #=======================================================================
        res_d = dict()
        for k,ar in ar_d.items():
            res_d[k] = self.write_array(ar, 
                       ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),
                       logger=log, **kwargs)
            
        #=======================================================================
        # wrap
        #=======================================================================
        return res_d
        
 
    def _check_dem_ar(self, ar):
        """check dem array satisfies assumptions"""
        #assert np.all(ar>0) #relaxing this
        assert np.all(~np.isnan(ar))
        assert 'float' in ar.dtype.name
 
    def _func_setup(self, dkey, 
                    logger=None, out_dir=None, tmp_dir=None,ofp=None,
 
                    write=None,layname=None,ext='.tif',
                    ):
        """common function default setup"""
        #logger
        if logger is None:
            logger = self.logger
        log = logger.getChild(dkey)
 
        
        #temporary directory
        if tmp_dir is None:
            tmp_dir = os.path.join(self.tmp_dir, dkey)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            
        #=======================================================================
        # #main outputs
        #=======================================================================
        if out_dir is None: out_dir = self.out_dir
        if not os.path.exists(out_dir):os.makedirs(out_dir)
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
        
        
class Haz(DownsampleChild, Session):
    """tools for experimenting with downsample sets"""
    
 
    
    def __init__(self, 

                 **kwargs):
        """
        
        Parameters
        ----------
 
 
        """
 
        
        super().__init__(obj_name='haz', wrk_dir=wrk_dir, **kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
 
        
    #===========================================================================
    # DOWNSAMPLING-----
    #===========================================================================
    def run_dsmp(self,demR_fp, wseR_fp,
 
                 dsc_l=None,
                 dscList_kwargs = dict(reso_iters=5),
                 
                 
                 method='direct',
                 out_dir=None,
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
 
        start = now()
        if out_dir is None: out_dir=os.path.join(self.out_dir, method)
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dsmp',  ext='.pkl', out_dir=out_dir, **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=tmp_dir, write=write)
        
        #=======================================================================
        # populate downsample set
        #=======================================================================
        if dsc_l is None:
            dsc_l = self.get_dscList(**dscList_kwargs, **skwargs)
            
        #=======================================================================
        # check divisibility
        #=======================================================================
        max_downscale = dsc_l[-1]
        if not is_divisible(demR_fp, max_downscale):
            log.warning('uneven division w/ %i... clipping'%max_downscale)
            
            dem_fp = self.build_crop(demR_fp, divisor=max_downscale, **skwargs)
            wse_fp = self.build_crop(wseR_fp, divisor=max_downscale, **skwargs)
            
        else:
            dem_fp, wse_fp = demR_fp, wseR_fp
 
            
        #=======================================================================
        # build the set from this
        #=======================================================================
        res_lib = self.build_dset(dem_fp, wse_fp, dsc_l=dsc_l, method=method, out_dir=out_dir)
        
        """not working
        self.build_vrts(res_lib, out_dir=os.path.join(out_dir, 'vrt'))"""
        #=======================================================================
        # build upscaled twins
        #=======================================================================
        #=======================================================================
        # res_libU = dict()
        # for downscale, fp_d in res_lib.items():
        #     res_libU[downscale] = self.build_upscales(fp_d, upscale=downscale)
        # log.info('upscaled %i'%len(res_libU))
        # #=======================================================================
        # # build vrts
        # #=======================================================================
        # self.build_vrts(res_libU)
        #=======================================================================
        
        log.info('finished in %.2f secs'%((now()-start).total_seconds()))
        
        #=======================================================================
        # assemble meta
        #=======================================================================
        #=======================================================================
        # meta_df = pd.DataFrame.from_dict(res_lib).T.join(
        #     pd.DataFrame.from_dict(res_libU).T, rsuffix='_ups'
        #     ).reset_index(drop=False).rename(columns={'index':'downscale'})
        #=======================================================================
        meta_df = pd.DataFrame.from_dict(res_lib).T.reset_index(drop=False).rename(columns={'index':'downscale'})
        #write the meta
        meta_df.to_pickle(ofp)
        log.info('wrote %s meta to \n    %s'%(str(meta_df.shape), ofp))
 
        return ofp
        
        
            
    def get_dscList(self,
 
                           reso_iters=3,
 
                           **kwargs):
        """get a fibonaci like sequence for downscale multipliers
        (NOT resolution)
        
        Parameters
        ---------

        base_resolution: int, optional
            base resolution from which to construct the dsc_l
            default: build from dem_fp
            
        Returns
        -------
        dsc_l: list
            sequence of ints for downscale. first is 1
            NOTE: every entry must be a divisor of the last entry (for cropping)
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dscList',  **kwargs)
 
        
        l = [1]
        for i in range(reso_iters-1):
            l.append(l[i]*2)
        log.info('got %i: %s'%(len(l), l))
        return l
    
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
            

    def _load_layers(self, dem_fp, wse_fp, reso_max=None, **kwargs):
        dem_ds = self._base_set(dem_fp, **kwargs)
        _ = self._base_inherit()
        dem_ar = load_array(dem_ds).astype(np.float32)
        self._check_dem_ar(dem_ar)
        wse_ds = self.open_dataset(wse_fp, **kwargs)
        wse_ar = load_array(wse_ds).astype(np.float32)
        
    #precheck
        assert_rlay_simple(dem_ds, msg='dem')
        assert_extent_equal(dem_ds, wse_ds, msg='dem vs wse')
        assert is_divisible(dem_ds, reso_max), 'passed DEM shape must be divisible by the max resolution (%i)' % dsc_l[-1]
        
        return wse_ar, dem_ar, dem_ds, wse_ds

    def build_dset(self,
            dem_fp, wse_fp,
            dsc_l=None,
            method='direct', resampleAlg='average',
            compress=None,
  
            out_dir=None,  
            **kwargs):
        """build a set of downsampled rasters
        
        Parameters
        ------------
        method: str default 'direct'
            downsample routine method
            
        compress: str, default 'compress'
            compression for outputs
            
        resampleAlg: str, default 'average'
            rasterio resampling method
            
        buildvrt: bool, default True
            construct a vrt of each intermittent raster (for animations)
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        start = now()
        log, tmp_dir, _, ofp, layname, write = self._func_setup('dsmp',  **kwargs)
        
        skwargs = dict(logger=log, write=write)
        
        #directories
        if out_dir is None: out_dir = os.path.join(self.out_dir, 'dset')
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        
        
        log.info('building %i downsamples from \n    %s\n    %s'%(
            len(dsc_l)-1, os.path.basename(dem_fp), os.path.basename(wse_fp)))
        
 
        
        #=======================================================================
        # open base layers
        #=======================================================================
        wse_ar, dem_ar, dem_ds, wse_ds = self._load_layers(dem_fp, wse_fp, reso_max=dsc_l[-1],**skwargs)
        
        base_resolution = int(dem_ds.res[0])
        log.info('base_resolution=%i, shape=%s' % (base_resolution, dem_ds.shape))
        
        #=======================================================================
        # build base depth
        #=======================================================================
        wd_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
        
        #create a datasource in memory        
        wd_ds = self.load_memDataset(wd_ar, **skwargs)
 
        
        base_ar_d = {'dem':dem_ar, 'wse':wse_ar, 'wd':wd_ar}
        base_ds_d = {'dem':dem_ds, 'wse':wse_ds, 'wd':wd_ds}
        #=======================================================================
        # loop and build downsamples
        #=======================================================================
        first=True
        res_lib=dict()
        for i, downscale in enumerate(dsc_l):
            log.info('    (%i/%i) reso=%i'%(i, len(dsc_l), downscale))
            
            with DownsampleChild(session=self,downscale=downscale, 
                                 crs=self.crs, nodata=self.nodata,transform=self.transform,
                                 compress=compress, out_dir=out_dir) as wrkr:
                
 
                #===================================================================
                # base/first
                #===================================================================
                """writing raw for consistency"""
                if first:
                    assert downscale==1
                    res_lib[downscale] = wrkr.write_dataset_d(base_ar_d, logger=log)
 
                    first = False
                    continue
                
                #===============================================================
                # downscale
                #===============================================================
                if method=='direct':
                    res_lib[downscale] = wrkr.downscale_direct(base_ds_d,resampleAlg=resampleAlg, **skwargs)
                elif method=='filter':
                    res_lib[downscale] = wrkr.downscale_filter(base_ds_d,resampleAlg=resampleAlg, **skwargs)
                else:
                    raise IOError('not implemented')
 
 
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i w/ method=%s in %.2f secs'%(
            len(res_lib), method, (now()-start).total_seconds()))
                 
            
        
        return res_lib
    

    
    #===========================================================================
    # COMPILING---------
    #===========================================================================
    def run_vrts(self, pick_fp, 
                 out_dir=None,
                 **kwargs):
        #log, tmp_dir, _, ofp, layname, write = self._func_setup('vrt',  **kwargs)
        
        """
        self.out_dir
        """
        meta_df = pd.read_pickle(pick_fp)
        
        #get ouptut directory in the same location as the data files
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(meta_df.iloc[0, :]['dem'])), 'vrt')
        
        return self.build_vrts(meta_df.set_index('downscale').to_dict(orient='index'), out_dir=out_dir, **kwargs)
        
    
    def build_vrts(self,res_lib,
                   out_dir=None,
                   **kwargs):
        """build vrts of the results for nice animations"""
 
        log, tmp_dir, _, ofp, layname, write = self._func_setup('vrt',  **kwargs)
        if out_dir is None: out_dir=os.path.join(self.out_dir, 'vrt')
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        from osgeo import gdal
        vrt_d = dict()
        
        """
        help(gdal.BuildVRT)
        gdal.BuildVRTOptions()
        help(gdal.BuildVRTOptions)
        """
        
        for sub_dkey, d in pd.DataFrame.from_dict(res_lib).to_dict(orient='index').items():
            log.debug(sub_dkey)
            ofp = os.path.join(out_dir, '%s_%i.vrt'%(sub_dkey, len(d)))
            gdal.BuildVRT(ofp, list(d.values()), separate=True, resolution='highest', resampleAlg='nearest')
            vrt_d[sub_dkey] = ofp
            
        log.info('wrote %i vrts\n%s'%(len(vrt_d),vrt_d))
        
        return vrt_d
 
    def build_upscales(self,
                       fp_d = dict(),
                       upscale=1,out_dir=None,
                       **kwargs):
        """construct a set of upscaled rasters"""
        #=======================================================================
        # defaults
        #=======================================================================
 
        log, tmp_dir, _, ofp, layname, write = self._func_setup('upsacle',  **kwargs)
        if out_dir is None: out_dir=os.path.join(self.out_dir, 'upscale', '%03i'%upscale)
        os.makedirs(out_dir)
        
        log.info('upscale=%i on %i'%(upscale, len(fp_d)))
        
        res_d = dict()
        for k, fp in fp_d.items():
            res_d[k] = resample(fp, os.path.join(out_dir, '%s_x%03i.tif'%(k, upscale)), scale=upscale)
        
        log.info('wrote %i to %s'%(len(res_d), out_dir))
        return res_d
    
    #===========================================================================
    # CASE MASKS---------
    #===========================================================================
    def run_catMask(self, pick_fp,
                    **kwargs):
        """build the dsmp cat mask for each reso iter"""
        
        log, tmp_dir, _, ofp, layname, write = self._func_setup('cMask',  **kwargs)
        skwargs = dict(out_dir=tmp_dir, logger=log)
        
        meta_df = pd.read_pickle(pick_fp)
        
        for i, row in meta_df.iterrows():
            dem_fp, wse_fp, downscale = row['dem'], row['wse'], row['downscale']
            #===================================================================
            # base/first
            #===================================================================
            if i==0:
                assert downscale==1
                demF_fp, wseF_fp = dem_fp, wse_fp
                #===============================================================
                # wse_ar, dem_ar, dem_ds, wse_ds = self._load_layers(dem_fp, wse_fp, 
                #                                            reso_max=meta_df.iloc[-1, 0],**skwargs)
                #===============================================================
                
                continue
 
            
            #===================================================================
            # classify
            #===================================================================
            raise IOError('stopped here')
            with DsampClassifier(session=self) as wrkr:
                wrkr.build_cat_masks(demF_fp,dem_fp, wseF_fp, downscale=downscale, **skwargs)
    
    #===========================================================================
    # STATS-------
    #===========================================================================
    def run_stats(self, pick_fp, 
                 out_dir=None,
                 **kwargs):
        """
        compute stats for each raster using each mask
        """
        log, tmp_dir, _, ofp, layname, write = self._func_setup('stats',  **kwargs)
        
        """
        self.out_dir
        """
        meta_df = pd.read_pickle(pick_fp)
 
    
    
