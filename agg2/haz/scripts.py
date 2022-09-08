'''
Created on Aug. 27, 2022

@author: cefect

aggregation hazard 
'''
import numpy as np
import numpy.ma as ma
import pandas as pd
import os, copy, datetime
import rasterio as rio
from rasterio.enums import Resampling
import scipy.ndimage

 
from definitions import wrk_dir
 
from hp.oop import Session
from hp.rio import RioWrkr, assert_extent_equal, is_divisible, assert_rlay_simple, load_array
from hp.basic import get_dict_str
from hp.pd import view
from hp.sklearn import get_null_confusion
from agg2.haz.rsc.scripts import ResampClassifier
from agg2.haz.misc import assert_dem_ar, assert_wse_ar
idx= pd.IndexSlice
#from skimage.transform import downscale_local_mean
#debugging rasters
#===============================================================================
# from hp.plot import plot_rast
# import matplotlib.pyplot as plt
#===============================================================================


def now():
    return datetime.datetime.now()


class UpsampleChild(ResampClassifier):
    """child for performing a single downsample set
    
    NOTES
    -------
    I thought it cleaner, and more generalizeable, to keep this on a separate worker"""
    
    def __init__(self, 
                 
                 subdir=True,
                 **kwargs):
 
        #=======================================================================
        # build defaults
        #=======================================================================
 
        super().__init__(subdir=subdir,
                          **kwargs)
        
    def agg_direct(self,
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
            
    def agg_filter(self,
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
        
 
class UpsampleSession(UpsampleChild, Session):
    """tools for experimenting with downsample sets"""
    
    def __init__(self, 

                 **kwargs):
        """
        
        Parameters
        ----------
 
 
        """
        
        super().__init__(obj_name='usmp', wrk_dir=wrk_dir,subdir=False, **kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
 
        print('finished __init__')
        
    #===========================================================================
    # UPSAMPLING (aggregating)-----
    #===========================================================================
    def run_agg(self,demR_fp, wseR_fp,
 
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
        #if out_dir is None: out_dir=os.path.join(self.out_dir, method)
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('agg',  ext='.pkl', subdir=True, **kwargs)
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
        meta_df['downscale'] =meta_df['downscale'].astype(int)  #already int
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

    def _load_datasets(self, dem_fp, wse_fp, divisor=None, **kwargs):
        """helper to load WSe and DEM datasets with some checks and base setting"""
        dem_ds = self._base_set(dem_fp, **kwargs)
        _ = self._base_inherit()

        wse_ds = self.open_dataset(wse_fp, **kwargs)
        
        #precheck
        assert_rlay_simple(dem_ds, msg='dem')
        assert_extent_equal(dem_ds, wse_ds, msg='dem vs wse')
        if not divisor is None:
            assert is_divisible(dem_ds, divisor), 'passed DEM shape not evenly divisible (%i)' % divisor
        
        return dem_ds, wse_ds

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
        dem_ds, wse_ds = self._load_datasets(dem_fp, wse_fp, divisor=dsc_l[-1],**skwargs)
        
        dem_ar = load_array(dem_ds)
        assert_dem_ar(dem_ar)
        
        wse_ar = load_array(wse_ds)
        assert_wse_ar(wse_ar)
        
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
            
            with UpsampleChild(session=self,downscale=downscale, 
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
                    res_lib[downscale] = wrkr.agg_direct(base_ds_d,resampleAlg=resampleAlg, **skwargs)
                elif method=='filter':
                    res_lib[downscale] = wrkr.agg_filter(base_ds_d,resampleAlg=resampleAlg, **skwargs)
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
                 cols = ['dem', 'wse', 'wd', 'catMosaic_fp'],
                 **kwargs):
        #log, tmp_dir, _, ofp, layname, write = self._func_setup('vrt',  **kwargs)
        icoln='downscale'
        """
        self.out_dir
        meta_df.columns
        """
        df_raw = pd.read_pickle(pick_fp) 
        
        #slice to specfied columns
        df = df_raw.loc[:, df_raw.columns.isin(cols+[icoln])]
        
        #=======================================================================
        # #get ouptut directory in the same location as the data files
        # if out_dir is None:
        #     out_dir = os.path.join(os.path.dirname(os.path.dirname(df.iloc[0, :]['dem'])), 'vrt')
        #=======================================================================
        
        return self.build_vrts(df.set_index(icoln).to_dict(orient='index'), out_dir=out_dir, **kwargs)
    
    def build_vrts(self,res_lib,
 
                   **kwargs):
        """build vrts of the results for nice animations"""
 
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('vrt', subdir=True, **kwargs)
 
        
        from osgeo import gdal
        vrt_d = dict()
        
        """
        help(gdal.BuildVRT)
        gdal.BuildVRTOptions()
        help(gdal.BuildVRTOptions)
        """
        
        for sub_dkey, d in pd.DataFrame.from_dict(res_lib).to_dict(orient='index').items():
            log.debug(sub_dkey)
            ofp = os.path.join(out_dir, '%s_%i.vrt'%(sub_dkey.replace('_fp', ''), len(d)))
            
            #pull reals
            fp_l = [k for k in d.values() if isinstance(k, str)]
            for k in fp_l: assert os.path.exists(k)
            
            gdal.BuildVRT(ofp, fp_l, separate=True, resolution='highest', resampleAlg='nearest')
            if os.path.exists(ofp):
                vrt_d[sub_dkey] = ofp
            else:
                raise IOError('failed to build vrt')
            
        log.info('wrote %i vrts\n%s'%(len(vrt_d),get_dict_str(vrt_d)))
        
        return vrt_d
 
 #==============================================================================
 #    def build_upscales(self,
 #                       fp_d = dict(),
 #                       upscale=1,out_dir=None,
 #                       **kwargs):
 #        """construct a set of upscaled rasters"""
 #        #=======================================================================
 #        # defaults
 #        #=======================================================================
 # 
 #        log, tmp_dir, _, ofp, layname, write = self._func_setup('upsacle',  **kwargs)
 #        if out_dir is None: out_dir=os.path.join(self.out_dir, 'upscale', '%03i'%upscale)
 #        os.makedirs(out_dir)
 #        
 #        log.info('upscale=%i on %i'%(upscale, len(fp_d)))
 #        
 #        res_d = dict()
 #        for k, fp in fp_d.items():
 #            res_d[k] = resample(fp, os.path.join(out_dir, '%s_x%03i.tif'%(k, upscale)), scale=upscale)
 #        
 #        log.info('wrote %i to %s'%(len(res_d), out_dir))
 #        return res_d
 #==============================================================================
    
    #===========================================================================
    # CASE MASKS---------
    #===========================================================================
    def run_catMasks(self, pick_fp,
                    **kwargs):
        """build the dsmp cat mask for each reso iter"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        start = now()
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('cMasks',subdir=True, ext='.pkl', **kwargs)
        
        dsmp_df = pd.read_pickle(pick_fp) #resuls from downsample
        
        #=======================================================================
        # build for each
        #=======================================================================
        res_d=dict()
        meta_lib = dict()
        ofp_d = dict()
        for i, row in dsmp_df.iterrows():
            #===================================================================
            # extract
            #===================================================================
            downscale = row['downscale']
            
            #===================================================================
            # defaults
            #===================================================================
            iname = '%03d'%downscale
            skwargs = dict(out_dir=tmp_dir, logger=log.getChild(iname), tmp_dir=tmp_dir)
            #===================================================================
            # base/first
            #===================================================================
            """categorization is always applied on the fine scale"""
            if i==0:
                assert downscale==1
                
                #===============================================================
                # load the base layers
                #===============================================================
                dem_fp, wse_fp = row['dem'], row['wse']
 
                dem_ds, wse_ds = self._load_datasets(dem_fp, wse_fp, reso_max=int(dsmp_df.iloc[-1, 0]),**skwargs)
                
                dem_ar = load_array(dem_ds)
        
                assert_dem_ar(dem_ar)
         
                assert_extent_equal(dem_ds, wse_ds)        
                
                wse_ar = load_array(wse_ds)
                assert_wse_ar(wse_ar)
                
                continue
 
            """
            wse_ds.read(1, masked=True)
            dem_ar = load_array(dem_ds)
            assert_dem_ar(dem_ar)
            
            wse_ar = load_array(wse_ds)
            assert_wse_ar(wse_ar)
                
            plot_rast(dem_ar)
            plot_rast(wse_ar, cmap='Blues')
            """
            #===================================================================
            # classify
            #===================================================================
            log.info('(%i/%i) downscale=%i building downsamp cat masks'%(i+1, len(dsmp_df), downscale)) 
            with ResampClassifier(session=self, downscale = downscale,  **skwargs) as wrkr:
                #build each mask
                cm_d = wrkr.get_catMasks(dem_ds=dem_ds, wse_ds=wse_ds, wse_ar=wse_ar, dem_ar=dem_ar)
                
                #build the mosaic
                cm_ar = wrkr.get_catMosaic(cm_d)
                
                #compute some stats
                stats_d = wrkr.get_catMasksStats(cm_d)
                
                #update
                res_d[downscale], meta_lib[downscale] = cm_ar, stats_d
                
            #===================================================================
            # write
            #===================================================================
            ofp_d[downscale] = self.write_array(cm_ar, logger=log,ofp=os.path.join(out_dir, 'catMosaic_%03i.tif'%downscale))
                
        log.info('finished building %i dsc mask mosaics'%len(res_d))
        #=======================================================================
        # #assemble meta
        #=======================================================================
        dx = pd.concat({k:pd.DataFrame.from_dict(v) for k,v in meta_lib.items()})
 
        #just the sum
        meta_df = dx.loc[idx[:, 'sum'], :].droplevel(1).astype(int)
        
        meta_df = dsmp_df.join(meta_df, on='downscale') 
 
        meta_df = meta_df.join(pd.Series(ofp_d).rename('catMosaic_fp'), on='downscale')
            
        #=======================================================================
        # write meta
        #=======================================================================
        meta_df.to_pickle(ofp)
        log.info('finished in %.2f secs and wrote %s to \n    %s'%((now()-start).total_seconds(), str(meta_df.shape), ofp))
        
        return ofp

    #===========================================================================
    # STATS-------
    #===========================================================================




    def _stat_wrap(self, res_lib, meta_d, ofp):
        res_dx = pd.concat(res_lib, axis=0, names=['scale', 'metric']).unstack()
    #ammend commons to index
        if not meta_d is None:
            mindex = pd.MultiIndex.from_frame(
                res_dx.index.to_frame().reset_index(drop=True).join(pd.DataFrame.from_dict(meta_d).T.astype(int), on='scale'))
            res_dx.index = mindex
        """
    
    view(res_dx)
    
    """
        res_dx.to_pickle(ofp)
        return res_dx

    def run_stats(self, pick_fp, 
 
                 cols = ['dem', 'wse', 'wd', 'catMosaic_fp'],
 
                 **kwargs):
        """
        compute global stats for each aggregated raster using each mask.
        mean(depth), sum(area), sum(volume)
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('stats',  subdir=True,ext='.pkl', **kwargs)
 
        start=now()
        icoln='downscale'
        
        #=======================================================================
        # load data
        #=======================================================================
        df_raw = pd.read_pickle(pick_fp) 
        
        #slice to specfied columns
        df = df_raw.loc[:, df_raw.columns.isin(cols+[icoln])].set_index(icoln)
        
        """
        view(df)
        """
        
        log.info('computing stats on %s'%str(df.shape))
        #=======================================================================
        # compute for each downscale
        #=======================================================================
        res_lib=dict()
        meta_d = dict()
        for i, row in df.iterrows():
            log.info('computing for downscale=%i'%i)
            #===================================================================
            # setup masks
            #===================================================================
            #the complete mask
            with rio.open(row['wd'], mode='r') as ds:
                shape = ds.shape                    
                mask_d = {'all':np.full(shape, True)}
 
                pixelArea = ds.res[0]*ds.res[1]
                pixelLength=ds.res[0]
                
                #load the array
                wd_ar = load_array(ds)
            
            #build other masks
            if i>1:
                cm_ar = load_array(row['catMosaic_fp'])
                
                """mosaics are stored at the native resolution
                here we decimate down to the downscaled resolution
                ... could consider storing these at the low resolution to speed things up"""
                cm_ar1 = cm_ar[::i, ::i] #decimate
 
                assert cm_ar1.shape==shape
                
                #boolean mask of each category
                mask_d.update(self.mosaic_to_masks(cm_ar1))
                
 
            #===================================================================
            # compute on each layer
            #===================================================================
            res_d1 = dict()
            for layName, ar_raw in {'wd': wd_ar}.items():
                """only doing wd for now"""
                #===================================================================
                # compute stats function on each mask
                #===================================================================
                
                func = lambda x:self.get_depth_stats(x, pixelArea=pixelArea)
                d = self.get_maskd_func(mask_d, ar_raw, func, log.getChild('%i.%s'%(i, layName)))
 
                res_d1[layName] = pd.DataFrame.from_dict(d)
            
 
            #===============================================================
            # store
            #===============================================================
            """
            view(pd.concat(res_d1, axis=1, names=['layer', 'dsc']))
            """
            res_lib[i] = pd.concat(res_d1, axis=1, names=['layer', 'dsc'])
            meta_d[i] = {'pixelArea':pixelArea, 'pixelLength':pixelLength}                    
 
            
        #=======================================================================
        # wrap
        #=======================================================================
        res_dx = self._stat_wrap(res_lib, meta_d, ofp)
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(res_dx.shape), ofp))
        
        return ofp
    
 
    def run_stats_fine(self, pick_fp, 
 
                 cols = ['wse', 'wd', 'catMosaic_fp'],
 
                 **kwargs):
        """
        compute global stats on fine rasters using cat masks
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('statsF',  subdir=True,ext='.pkl', **kwargs)
 
        start=now()
        icoln='downscale'
        
        #=======================================================================
        # load data
        #=======================================================================
        df_raw = pd.read_pickle(pick_fp) 
        
        #slice to specfied columns
        df = df_raw.loc[:, df_raw.columns.isin(cols+[icoln])].set_index(icoln)
        
        log.info('computing stats on %s'%str(df.shape))
        #=======================================================================
        # compute for each downscale
        #=======================================================================
        res_lib, meta_d=dict(), dict()
        for i, row in df.iterrows():
            log.info('computing for downscale=%i'%i)
            #===================================================================
            # load baseline
            #===================================================================
            if i==1:
                #the complete mask
                with rio.open(row['wd'], mode='r') as ds:
                    #get baseline data
                    
                    wdF_ar = load_array(ds)                    
                    shape = ds.shape       
                    
                    #build for this loop
                    mask_d = {'all':np.full(shape, True)}
     
                    pixelArea = ds.res[0]*ds.res[1]
                    pixelLength=ds.res[0]
                
 
            #===================================================================
            # #build other masks
            #===================================================================
            if i>1:
                cm_ar = load_array(row['catMosaic_fp'])
                
                """here wee keep the fine resolution"""
                assert cm_ar.shape==wdF_ar.shape
                
                mask_d.update(self.mosaic_to_masks(cm_ar))
   
            #===================================================================
            # compute on each layer
            #===================================================================
            res_d1 = dict()
            for layName, ar_raw in {'wd': wdF_ar}.items():
                """only doing wd for now
                unlike the upscaled version.. here we always compute against the fine wd"""
                #===================================================================
                # compute stats function on each mask
                #===================================================================
                
                func = lambda x:self.get_depth_stats(x, pixelArea=pixelArea)
                d = self.get_maskd_func(mask_d, ar_raw, func, log.getChild('%i.%s'%(i, layName)))
 
                res_d1[layName] = pd.DataFrame.from_dict(d)
                
            #===============================================================
            # store
            #===============================================================
            """
            view(pd.concat(res_d1, axis=1, names=['layer', 'dsc']))
            """
            res_lib[i] = pd.concat(res_d1, axis=1, names=['layer', 'dsc'])
            #meta_d[i] = {'pixelArea':pixelArea, 'pixelLength':pixelLength} 
            
        #=======================================================================
        # wrap
        #=======================================================================
        """
        view(res_dx)
        """
        res_dx = self._stat_wrap(res_lib, None, ofp)
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(res_dx.shape), ofp))
        
        return ofp
        
        
    def get_depth_stats(self, mar, pixelArea):
        res_d=dict()
        #=======================================================
    # simple mean
    #=======================================================
        res_d['mean'] = mar.mean()
    #===================================================================
    # inundation area
    #===================================================================
        res_d['posi_area'] = np.sum(mar>0) * (pixelArea) #non-nulls times pixel area
    #===================================================================
    # volume
    #===================================================================
        res_d['vol'] = mar.sum() * pixelArea
        
        return res_d


    def get_maskd_func(self, mask_d, ar_raw, func, log):
        log.debug('    on %s w/ %i masks'%(str(ar_raw.shape), len(mask_d)))
        res_lib = dict()
        for maskName, mask_ar in mask_d.items():
            log.info('     %s (%i/%i)' % (maskName, mask_ar.sum(), mask_ar.size))
            res_d = {'count':mask_ar.sum()}
 
            #===============================================================
            # some valid cells
            #===============================================================
            if np.any(mask_ar):
                #build masked
                mar = ma.array(ar_raw, mask=~mask_ar) #valids=True
                res_d.update(func(mar))
            else:
                log.warning('%s got no valids' % (maskName))
                #res_d.update({'wd_mean':0.0, 'wse_area':0.0, 'vol':0.0})
            res_lib[maskName] = res_d #store
        
        log.debug('    finished on %i masks'%len(res_lib))
        return res_lib
        
    #===========================================================================
    # ERRORS-------
    #===========================================================================
    def run_errs(self,pick_fp, **kwargs):
        """build difference grids for each layer"""
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('errs',  subdir=True,ext='.pkl', **kwargs)
        start = now()
        #=======================================================================
        # load
        #=======================================================================
        df_raw = pd.read_pickle(pick_fp).set_index('downscale')
        
        log.info('on %i'%len(df_raw))
        #=======================================================================
        # loop on each layer
        #=======================================================================
        res_lib = dict()
        for layName in ['wse']:
            #===================================================================
            # defaults
            #===================================================================
            fp = df_raw.loc[1, layName]
            log.info('for \'%s\' from %s'%(layName, os.path.basename(fp)))
            
            res_cm_d, res_d = dict(), dict()
            #===================================================================
            # #load baseline
            #===================================================================
            with rio.open(fp, mode='r') as ds: 
                assert ds.res[0]==1
                base_ar = ds.read(1, masked=True)
                self._base_inherit(ds=ds)
                
            #===================================================================
            # loop on reso
            #===================================================================
            for i, (scale, fp) in enumerate(df_raw[layName].items()):
                log.info('    %i/%i scale=%i from %s'%(i+1, len(df_raw), scale, os.path.basename(fp)))
 
                #===============================================================
                # vs. base (no error)
                #===============================================================
                if i==0:
 
                    res_ar = ma.masked_where(base_ar.mask, np.full(base_ar.shape, 0)) #same mask as base w/ some zeros
                    cm_ser = pd.Series(
                        {'TP':np.isnan(base_ar).sum(), 'FP':0, 'TN':base_ar.size-np.isnan(base_ar).sum(), 'FN':0},
                        dtype=int)
                    
                #===============================================================
                # vs. an  upscale
                #===============================================================
                else:
                    #get disagg
                    with rio.open(fp, mode='r') as ds:
                        fine_ar = ds.read(1, out_shape=base_ar.shape, resampling=Resampling.nearest, masked=True)
                        #resample load
                        #=======================================================
                        # resamp_kwargs = dict(out_shape=base_ar.shape, resampling=Resampling.nearest)
                        # fine_raw_ar = ds.read(1, **resamp_kwargs)                        
                        # fine_mask = ds.read_masks(1,**resamp_kwargs)
                        # 
                        # #handle nulls
                        # fine_ar = np.where(fine_mask==0,  ds.nodata, fine_raw_ar).astype(np.float32)
                        #=======================================================
                        #=======================================================
                        # coarse_ar = load_array(ds) 
                        # fine_ar = scipy.ndimage.zoom(coarse_ar, scale, order=0, mode='reflect',   grid_mode=True)
                        #=======================================================
                        assert fine_ar.shape==base_ar.shape
                        
                    #compute errors
                    res_ar = fine_ar - base_ar
                    
                    #compute null confusion
                    cm_df = get_null_confusion(base_ar, fine_ar, names=['fine', 'base'])                    
                    cm_ser = cm_df.set_index('codes')['counts']
                    
                    log.info('    calcd: %s'%cm_ser.to_dict())
                    

                #===============================================================
                # write
                #===============================================================
                assert isinstance(res_ar, ma.MaskedArray)
                assert not np.all(np.isnan(res_ar))
                assert not np.all(res_ar.mask)
 
                od = os.path.join(out_dir, layName)
                if not os.path.exists(od):os.makedirs(od)
                res_d[scale] = self.write_array(res_ar, ofp=os.path.join(od, '%s_err_%03i.tif'%(layName, scale)), 
                                                logger=log, masked=True)
                
                #===============================================================
                # wrap scale
                #===============================================================
                res_cm_d[scale] = cm_ser
                
            #===================================================================
            # wrap lyaer
            #===================================================================
 
            res_lib[layName] = pd.concat(res_cm_d, axis=1).T.join(pd.Series(res_d).rename('err_fp'))
 
            
            
        #=======================================================================
        # wrap on all
        #=======================================================================
        """as we are adding a nother level, need to do some re-organizing of the inputs"""
        #collect
        res_dx1 = pd.concat(res_lib,axis=1, names=['layer', 'val'])
        res_dx1.index.name=df_raw.index.name
        
        #join with input filepaths
        res_dx = pd.concat({'fp':df_raw.loc[:, ['dem', 'wse', 'wd', 'catMosaic_fp']]}, axis=1, names=['val', 'layer']).swaplevel(axis=1).join(res_dx1).sort_index(axis=1)
        
        #rename catMosaic level values\
        res_dx = res_dx.rename(columns={'catMosaic_fp':'catMosaic'})        
        
        #join the dsc
        res_dx = res_dx.join(pd.concat({'catMosaic':df_raw.loc[:, ['DD', 'WW', 'WP', 'DP']]}, axis=1, names=['layer', 'val'])).sort_index(axis=1)
 
        """
        view(df_raw)
        view(res_dx.T)
        view(pd.concat(res_lib,axis=1))
        """
        #=======================================================================
        # #write
        #=======================================================================
        res_dx.to_pickle(ofp)
        
        log.info('finished on %s in %.2f secs and wrote to\n    %s'%(str(res_dx.shape), (now()-start).total_seconds(), ofp))
        
        return ofp
    
    

        """
        view(res_dx)
        np.isnan(base_ar).sum()
        view(cm_dx.unstack())
        """
        
        
    def run_errStats(self,pick_fp, **kwargs):
        """compute stats from diff rasters"""
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('errStats',  subdir=True,ext='.pkl', **kwargs)
        start = now()
        
        dxcol_raw = pd.read_pickle(pick_fp)
        confusion_l = ['FN', 'FP', 'TN', 'TP']
        #=======================================================================
        # loop on each layer
        #=======================================================================
        res_lib=dict()
        for layName in ['wse']:
            log.info('on \'%s\''%layName)
            df_raw = dxcol_raw[layName]
            assert 'err_fp' in df_raw.columns
            
            #===================================================================
            # loop on each scale
            #===================================================================
            res_d1 = dict()
            for i, (scale, row) in enumerate(df_raw.iterrows()):
                log.info('    %i/%i from %s'%(i+1, len(df_raw), os.path.basename(row['err_fp'])))
                res_d = dict()
            

                    
                #===============================================================
                # compute metrics
                #===============================================================
                if i==0:
                    res_d.update({'meanErr':0.0, 'meanAbsErr':0.0, 'RMSE':0.0})
                else:
                    with rio.open(row['err_fp'], mode='r') as ds:
                        ar = ds.read(1, masked=True)
                        
                    assert not np.all(np.isnan(ar)), scale
                    res_d['meanErr'] = ar.sum()/ar.size
                    res_d['meanAbsErr'] = np.abs(ar).sum()/ar.size
                    res_d['RMSE'] = np.sqrt(np.mean(np.square(ar)))
                
                    del ar
                
                #===============================================================
                # add confusion
                #===============================================================
                res_d.update(row.loc[confusion_l].to_dict())
                
                #===============================================================
                # wrap
                #===============================================================
                res_d1[scale] = res_d
            #===================================================================
            # wrap layer loop
            #===================================================================
            res_lib[layName] = pd.DataFrame.from_dict(res_d1).T.astype({k:np.int32 for k in confusion_l})
        #=======================================================================
        # wrap on layer
        #=======================================================================
        res_dx = pd.concat(res_lib, axis=1, names=['layer', 'metric'])
        
        #=======================================================================
        # #write
        #=======================================================================
        res_dx.to_pickle(ofp)
        
        log.info('finished on %s in %.2f secs and wrote to\n    %s'%(str(res_dx.shape), (now()-start).total_seconds(), ofp))
        
        return ofp
            
        
    
    
