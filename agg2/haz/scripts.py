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
#import scipy.ndimage

from osgeo import gdal
from sklearn.metrics import confusion_matrix
 

from hp.rio import RioWrkr, assert_extent_equal, is_divisible, assert_rlay_simple, load_array, \
    assert_ds_attribute_match
from hp.basic import get_dict_str
from hp.pd import view
from hp.sklearn import get_confusion

from agg2.coms import Agg2Session, AggBase
from agg2.haz.rsc.scripts import ResampClassifier
from agg2.haz.coms import assert_dem_ar, assert_wse_ar, assert_dx_names, index_names, coldx_d
idx= pd.IndexSlice
#from skimage.transform import downscale_local_mean
#debugging rasters
#===============================================================================
# from hp.plot import plot_rast
# import matplotlib.pyplot as plt
#===============================================================================
from hp.plot import plot_rast #for debugging

#import dask.array as da

def now():
    return datetime.datetime.now()


class UpsampleChild(ResampClassifier, AggBase):
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
 
        super().__init__(subdir=subdir,**kwargs)
        

        
    def agg_direct(self,
                         ds_d,
                         resampleAlg='average',
                         downscale=None,
                         **kwargs):
        """direct aggregation of DEM and WD. WSE is recomputed
        
        NOTE
        ----------
         from DEM and WD (not a direct average of WSEs1)
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, layname, write = self._func_setup('direct',  **kwargs)
        if downscale is None: downscale=self.downscale
        start = now()
        #=======================================================================
        # downscale DEM an WD each
        #=======================================================================
        log.info('downscale=%i on %s'%(downscale, list(ds_d.keys())))
        res_d, ar_d = dict(), dict()
        for k, raw_ds in ds_d.items():
            if k=='wse':continue
            ds1 = self.resample(dataset=raw_ds, resampling=getattr(rio.enums.Resampling, resampleAlg), scale=1/downscale)
            
            #load array (for wse calc)
            ar_d[k] = ds1.read(1, masked=False)
            
            #write it
            res_d[k] = self.write_memDataset(ds1, dtype=np.float32,
                       ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),masked=True,
                       logger=log)
            
        #=======================================================================
        # compute WSE
        #=======================================================================
        k='wse'
        #wse_ar = ma.array(ar_d['dem'] + ar_d['wd'], mask=ar_d['wd']<=0, fill_value=ds1.nodata)
        wse_ar = np.where(ar_d['wd']<=0, ds1.nodata, ar_d['dem'] + ar_d['wd']).astype(np.float32)
        
        del ar_d
        
        res_d[k] = self.write_array(wse_ar,  masked=False, ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),
                               logger=log, nodata=ds1.nodata,
                               transform=ds1.transform, #use the resampled from above
                               )
            
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
        """fitlered agg of DEM and WSE. WD is recomputed."""
        
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
    
class RasterArrayStats(AggBase):

 
    def __init__(self,
                 engine='np', 
                 **kwargs):
        """methods for ocmputing raster stats on arrays
        
        Parameters
        ----------
        engine: str, default 'np'
            whether to use dask or numpy
        """
 
        #=======================================================================
        # build defaults
        #=======================================================================
        
 
        super().__init__(**kwargs)
        
        self.engine=engine
        
        
    def _build_statFuncs(self, engine=None):
        """construct the dictinoary of methods"""
        if engine is None: engine=self.engine
        
        if engine=='np':
            d = {'wse':lambda ar, **kwargs:self._get_wse_stats(ar, **kwargs),
                 'wd':lambda ar, **kwargs:self._get_depth_stats(ar, **kwargs),
                 'diff':lambda ar, **kwargs:self._get_diff_stats(ar, **kwargs),
                }
            
        else:
            raise IOError('not implemented')
        
        #check
        miss_l = set(d.keys()).difference(coldx_d['layer'])
        assert len(miss_l)==0, miss_l
        
        self.statFunc_d = d
            
        
        
    #===========================================================================
    # WSE--------
    #===========================================================================
    def _get_wse_stats(self, mar, **kwargs):
        
        assert_wse_ar(mar, masked=True)
        return {'mean': mar.mean()}
 
    

    #===========================================================================
    # WD-------
    #===========================================================================
    def _get_depth_stats(self, mar, pixelArea=None):
 
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

    def _get_depth_stats_dask(self, mar, pixelArea=None):
 
        res_d=dict()
        dar = da.from_array(mar, chunks='auto')
        #=======================================================
        # simple mean
        #=======================================================
        res_d['mean'] = dar.mean().compute() #mar.mean()
        
        
        #===================================================================
        # inundation area
        #===================================================================
        res_d['posi_area'] = (np.sum(dar>0) * (pixelArea)).compute() #non-nulls times pixel area
        #===================================================================
        # volume
        #===================================================================
        res_d['vol'] = (dar.sum() * pixelArea).compute()
        
        return res_d
    
    #===========================================================================
    # DIFFERENCE-------
    #===========================================================================
    def _get_diff_stats(self, ar):
        """compute stats on difference grids.
        NOTE: always using reals for denometer"""
        assert isinstance(ar, ma.MaskedArray)
        assert not np.any(np.isnan(ar))
        
        
        
        #fully masked check
        if np.all(ar.mask):
            return {'meanErr':0.0, 'meanAbsErr':0.0, 'RMSE':0.0}
        
 
        res_d = dict()
        rcnt = (~ar.mask).sum()
        res_d['sum'] = ar.sum()
        res_d['meanErr'] =  res_d['sum']/rcnt #same as np.mean(ar)
        res_d['meanAbsErr'] = np.abs(ar).sum() / rcnt
        res_d['RMSE'] = np.sqrt(np.mean(np.square(ar)))
        return res_d
    
    def _get_diff_stats_dask(self, ar, **kwargs):
        """compute stats on difference grids.
        NOTE: always using reals for denometer"""
        assert isinstance(ar, ma.MaskedArray)
        assert not np.any(np.isnan(ar))
        
        
        
        #fully masked check
        if np.all(ar.mask):
            return {'meanErr':0.0, 'meanAbsErr':0.0, 'RMSE':0.0}
        
        dar = da.from_array(ar, chunks='auto')
        res_d = dict()
        rcnt = (~ar.mask).sum()
        
        res_d['meanErr'] =  dar.sum().compute()/rcnt #same as np.mean(ar)
        res_d['meanAbsErr'] = np.abs(dar).sum().compute() / rcnt
        res_d['RMSE'] = np.sqrt(np.mean(np.square(dar))).compute()
        return res_d
        
 
class UpsampleSession(Agg2Session, RasterArrayStats, UpsampleChild):
    """tools for experimenting with downsample sets"""
    
    def __init__(self,method='direct', scen_name=None, obj_name='haz',**kwargs):
 
        if scen_name is None: scen_name=method
        super().__init__(obj_name=obj_name,scen_name=scen_name, **kwargs)
        self.method=method
        
 
        
    #===========================================================================
    # UPSAMPLING (aggregating)-----
    #===========================================================================
    def run_agg(self,demR_fp, wseR_fp,
 
                 dsc_l=None,
                 dscList_kwargs = dict(reso_iters=5),
                 
                 method=None,
 
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
        if method is None: method=self.method
        start = now()
        #if out_dir is None: out_dir=os.path.join(self.out_dir, method)
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('agg',  ext='.pkl', subdir=True, **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=tmp_dir, write=write)
        
        log.info('for %i upscales using \'%s\' from \n    DEM:  %s\n    WSE:  %s'%(
            len(dsc_l),method, os.path.basename(demR_fp), os.path.basename(wseR_fp)))
        #=======================================================================
        # populate downsample set
        #=======================================================================
        if dsc_l is None:
            dsc_l = self.get_dscList(**dscList_kwargs, **skwargs)
            
        #=======================================================================
        # check layers
        #=======================================================================
        for layName, fp in {'dem':demR_fp, 'wse':wseR_fp}.items():
            assert_ds_attribute_match(fp, crs=self.crs, nodata=self.nodata, msg=layName) 
            
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
        
 
 
        
        log.info('finished in %.2f secs'%((now()-start).total_seconds()))
        
        #=======================================================================
        # assemble meta
        #=======================================================================
 
        meta_df = pd.DataFrame.from_dict(res_lib).T.rename_axis('scale')
 
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
                if i==0:
                    assert downscale==1
                    res_lib[downscale] = wrkr.write_dataset_d(base_ar_d, logger=log) 
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
 
                 #cols = ['dem', 'wse', 'wd', 'catMosaic'],
                 **kwargs):
        """this really isnt working well... should find a different format"""
        log, tmp_dir, out_dir, ofp, layname0, write = self._func_setup('vrt',  subdir=True, **kwargs)
 
        """
        view(dxcol_raw)
        self.out_dir
        meta_df.columns
        """
        df = pd.read_pickle(pick_fp)
        log.info('compiling \'%s\' vrt from %s'%(layname0, os.path.basename(pick_fp))) 
        res_d = dict()
        
        for layName, col in df.items():  
            if not layName in coldx_d['layer']:
                continue
            #if not layName=='wse': continue 
            fp_d = col.dropna().to_dict()
            
            
            try:
                ofpi = self.build_vrts(fp_d,ofp = os.path.join(out_dir, '%s_%s_%i.vrt'%(layname0, layName,  len(fp_d))))
                
                log.info('    for \'%s\' compiled %i into a vrt: %s'%(layName, len(fp_d), os.path.basename(ofpi)))
                
                res_d['%s'%(layName)] = ofpi
            except Exception as e:
                log.error('failed to build vrt on %s w/ \n    %s'%(layName, e))
        #=======================================================================
        # for layer, gdx in dxcol_raw.groupby(level=0, axis=1):
        #     for coln, col in gdx.droplevel(0, axis=1).items():
        #         if not coln.endswith('fp'):
        #             continue
        #         fp_d = col.dropna().to_dict()
        #         ofpi = self.build_vrts(fp_d,ofp = os.path.join(out_dir, '%s_%s_%i.vrt'%(layer, coln, len(fp_d))))
        #         
        #         log.info('    for \'%s.%s\' compiled %i into a vrt: %s'%(layer, coln, len(fp_d), os.path.basename(ofpi)))
        #         
        #         res_d['%s_%s'%(layer, coln)] = ofpi
        #=======================================================================
        
        log.info('finished writing %i to \n    %s'%(len(res_d), out_dir))
        
        return res_d
    
    def build_vrts(self,fp_d,ofp):
        """build vrts of the results for nice animations"""
 
        #log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('b', subdir=False, **kwargs)
 
        """
        help(gdal.BuildVRT)
        gdal.BuildVRTOptions()
        help(gdal.BuildVRTOptions)
        """
 
        #ofp = os.path.join(out_dir, '%s_%i.vrt'%(sub_dkey.replace('_fp', ''), len(d)))
        
        #pull reals
        fp_l = [k for k in fp_d.values() if isinstance(k, str)]
        for k in fp_l: assert os.path.exists(k), k
        
        gdal.BuildVRT(ofp, fp_l, separate=True, resolution='highest', resampleAlg='nearest')
        
        if not os.path.exists(ofp): 
            raise IOError('failed to build vrt')
            

        
        return ofp
 
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
        for i, (downscale, row) in enumerate(dsmp_df.iterrows()):
 
            
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
 
                dem_ds, wse_ds = self._load_datasets(dem_fp, wse_fp, reso_max=int(dsmp_df.index[-1]),**skwargs)
                
                dem_ar = load_array(dem_ds, masked=True)        
                assert_dem_ar(dem_ar, masked=True)         
                
                wse_ar = load_array(wse_ds, masked=True)
                assert_wse_ar(wse_ar, masked=True)
                
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
                cm_d = wrkr.get_catMasks2(wse_ar=wse_ar, dem_ar=dem_ar)
                
                #build the mosaic
                cm_ar = wrkr.get_catMosaic(cm_d)
                
                #compute some stats
                stats_d = wrkr.get_catMasksStats(cm_d)
                
                #update
                res_d[downscale], meta_lib[downscale] = cm_ar, stats_d
                
            #===================================================================
            # write
            #===================================================================
            #new transform
            transform = dem_ds.transform * dem_ds.transform.scale(
                        (dem_ds.width / cm_ar.shape[-1]),
                        (dem_ds.height / cm_ar.shape[-2])
                    )
                    
            ofp_d[downscale] = self.write_array(cm_ar, logger=log,ofp=os.path.join(out_dir, 'catMosaic_%03i.tif'%downscale), transform=transform)
                
        log.info('finished building %i dsc mask mosaics'%len(res_d))
        #=======================================================================
        # #assemble meta
        #=======================================================================
        dx = pd.concat({k:pd.DataFrame.from_dict(v) for k,v in meta_lib.items()})
 
        #just the sum
        meta_df = dx.loc[idx[:, 'sum'], :].droplevel(1).astype(int).rename_axis(dsmp_df.index.name)
        
        #meta_df = dsmp_df.join(meta_df, on='scale') 
 
        meta_df = meta_df.join(pd.Series(ofp_d).rename('fp'), on=dsmp_df.index.name)
            
        #=======================================================================
        # write meta
        #=======================================================================
        meta_df.to_pickle(ofp)
        log.info('finished in %.2f secs and wrote %s to \n    %s'%((now()-start).total_seconds(), str(meta_df.shape), ofp))
        
        return ofp

    #===========================================================================
    # STATS-------
    #===========================================================================
 
    def run_stats(self, agg_fp, cm_fp,
 
                 layName_l = ['wse', 'wd'],
 
 
                 **kwargs):
        """
        compute global stats for each aggregated raster using each mask.
        mean(depth), sum(area), sum(volume)
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('stats',  subdir=True,ext='.pkl', **kwargs)
 
        df, start = self._rstats_init(agg_fp, cm_fp, layName_l, log)
        
        res_lib, meta_d =dict(), dict()
        #=======================================================================
        # compute for each scale
        #=======================================================================        
        for i, (scale, row) in enumerate(df.iterrows()):
            log.info(f'    {i+1}/{len(df)} on scale={scale}')
 
            #the complete mask
            with rio.open(row['wd'], mode='r') as ds:
                shape = ds.shape                    
                mask_d = {'all':np.full(shape, True)}
 
                pixelArea = ds.res[0]*ds.res[1]
                pixelLength=ds.res[0]
                
 
            fkwargs = dict(pixelArea=pixelArea)
            #build other masks
            if i>0:
                cm_ar = load_array(row['catMosaic']) 
                assert cm_ar.shape==shape
                
                #boolean mask of each category
                mask_d.update(self.mosaic_to_masks(cm_ar))
                
 
            #===================================================================
            # compute on each layer
            #===================================================================
            res_d1 = dict()
            for layName, fp in row.items():
                if layName=='catMosaic': continue
                logi = log.getChild('%i.%s'%(i, layName))
                
                #===============================================================
                # get stats func
                #===============================================================
                func = self.statFunc_d[layName]
                
                #===============================================================
                # load and compute
                #===============================================================
                log.debug('loading from %s'%fp)
                with rio.open(fp, mode='r') as ds:
                    ar_raw = load_array(ds, masked=True)                
 
                    d = self.get_maskd_func(mask_d, ar_raw, func, logi, **fkwargs)
 
                res_d1[layName] = pd.DataFrame.from_dict(d)
            
 
            #===============================================================
            # store
            #===============================================================
            """
            view(res_dx)
            """
            res_lib[scale] = pd.concat(res_d1, axis=1, names=['layer', 'dsc'])
            meta_d[scale] = {'pixelArea':pixelArea, 'pixelLength':pixelLength}                    
 
            
        #=======================================================================
        # wrap
        #=======================================================================
        res_dx = self._rstats_wrap(res_lib, meta_d, ofp)
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(res_dx.shape), ofp))
        
        return ofp
    
 
    def run_stats_fine(self, agg_fp, cm_fp, 
 
                 layName_l = ['wse', 'wd'],
 
 
                 **kwargs):
        """
        compute global stats on fine/raw rasters using cat masks
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('statsF',  subdir=True,ext='.pkl', **kwargs)
 
        df, start = self._rstats_init(agg_fp, cm_fp, layName_l, log)
        
        meta_d =dict()
        
        """because we loop in a differnet order from normal stats... need to precosntruct to match"""
        res_lib = {scale:{layName:dict() for layName in layName_l} for scale in df.index}
        #=======================================================================
        # compute for each layer
        #=======================================================================
        """runing different masks against the same layer... need a different outer loop"""
        for layName, fp in df.loc[1, layName_l].to_dict().items(): 
            #===================================================================
            # #load baseline layer
            #===================================================================
            with rio.open(fp, mode='r') as ds:
                #get baseline data
                
                ar_raw = load_array(ds, masked=True)                    
                shape = ds.shape
                                
                #scale info
                pixelArea = ds.res[0]*ds.res[1] #needed for scaler below
                
                metaF_d = get_pixel_info(ds)
                

            #===================================================================
            # loop on scale
            #===================================================================
            log.info('for \'%s\' w/ %s computing against %i scales'%(layName, str(ar_raw.shape), len(df)))
            
            
            #stat vars for this layer
            mask_full = np.full(shape, True)
            func = self.statFunc_d[layName]
            fkwargs = dict(pixelArea=pixelArea)
            

            #loop
            for i, (scale, row) in enumerate(df.iterrows()):
                #setup this scale
                logi = log.getChild('%i.%s'%(scale, layName))
                logi.debug(row.to_dict())
                
 
 
                def upd_meta(ds):
                    meta_d[scale] = get_pixel_info(ds)
                    
                #===============================================================
                # build baseline mask
                #===============================================================
                mask_d = {'all':mask_full} 
                if i==0:
                    meta_d[scale] = copy.deepcopy(metaF_d)
     
                #===================================================================
                # add other masks
                #===================================================================
                else:
                    """here we need to do wnscale"""
                    with rio.open(row['catMosaic'], mode='r') as ds:
                        cm_ar = ds.read(1, out_shape=shape, resampling=Resampling.nearest, masked=False)                        
                        upd_meta(ds)     
     
                    assert cm_ar.shape==shape
                    
                    mask_d.update(self.mosaic_to_masks(cm_ar))
                    
                    
                #===================================================================
                # compute stats function on each mask
                #=================================================================== 
                d = self.get_maskd_func(mask_d, ar_raw, func, logi, **fkwargs)
 
                res_lib[scale][layName] = pd.DataFrame.from_dict(d) 
                
            #===============================================================
            # wrap layer
            #===============================================================
            """handling this in the below reorient"""
 
            
        #=======================================================================
        # wrap
        #=======================================================================
        #re-orient container to match stats expectations
        d = dict()
        for scale, d1 in res_lib.items():
            d[scale] = pd.concat({l:df for l,df in d1.items()}, axis=1, names=['layer', 'dsc'])
        """
        view(res_dx)
        """
        res_dx = self._rstats_wrap(d, meta_d, ofp)
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(res_dx.shape), ofp))
        
        return ofp
        
        


    def get_maskd_func(self, mask_d, ar_raw, func, log, **kwargs):
        """apply each mask to the array then feed the result to the function"""
        
        log.debug('    on %s w/ %i masks'%(str(ar_raw.shape), len(mask_d)))
        res_lib = dict()
        
        for maskName, mask_ar in mask_d.items():
            """
            mask_ar=True: values we are interested in (opposite of numpy's mask convention)
            """
            log.info('     %s (%i/%i) on %s' % (maskName, mask_ar.sum(), mask_ar.size, str(mask_ar.shape)))
            res_d = {'pre_count':mask_ar.sum()}
            assert mask_ar.shape==ar_raw.shape
            #===============================================================
            # construct masked array
            #===============================================================
            mar = ma.array(1, mask=True) #dummy all invalid
            if np.any(mask_ar):
                #===============================================================
                # apply the mask
                #===============================================================
                #update an existing mask
                if isinstance(ar_raw, ma.MaskedArray):
                    #ar_raw.harden_mask() #make sure we don't unmask anything
                    mar = ar_raw.copy()
                    
                    if not np.all(mask_ar):                        
                        mar[~mask_ar] = ma.masked 
 
                    
                #construct mask from scratch 
                else:
                    mar = ma.array(ar_raw, mask=~mask_ar) #valids=True
                    
            #===============================================================
            # #execute the stats function
            #===============================================================
            res_d['post_count'] = (~mar.mask).sum()
            
            if res_d['post_count']>0:
                if  __debug__:
                    log.debug('    %i/%i valids and kwargs: %s'%(
                        res_d['post_count'], mar.size, kwargs))
                    
                res_d.update(func(mar, **kwargs))
            
            #===================================================================
            # everything invalid
            #===================================================================
            else:
                log.warning('%s got no valids' % (maskName))
            
            #===================================================================
            # wrap
            #=================================================================== 
            res_lib[maskName] = res_d #store
        
        log.debug('    finished on %i masks'%len(res_lib))
        return res_lib
    
    def _rstats_wrap(self, res_lib, meta_d, ofp):
        """post for rstat funcs
        Parameters
        -----------
        res_lib, dict
            {scale:dxcol (layer, dsc)}
        """
 
        res_dx = pd.concat(res_lib, axis=0, names=['scale', 'metric']).unstack()
        
        #ammend commons to index
        if not meta_d is None:
            mindex = pd.MultiIndex.from_frame(
                res_dx.index.to_frame().reset_index(drop=True).join(pd.DataFrame.from_dict(meta_d).T.astype(int), on='scale'))
            res_dx.index = mindex
            
        #sort and clean
        res_dx = res_dx.sort_index(axis=1, sort_remaining=True).dropna(axis=1, how='all')
        
 
        #checks
        assert not res_dx.isna().all().all()
        assert_dx_names(res_dx)
        
        #write
        res_dx.to_pickle(ofp)
        return res_dx
 


    def _rstats_init(self, agg_fp, cm_fp, layName_l, log):
        """init for rstat funcs"""
        start = now()
        icoln = 'downscale'
        self._build_statFuncs()
        
        #=======================================================================
        # load data
        #=======================================================================
        #run_agg
        df_raw = pd.read_pickle(agg_fp).loc[:, layName_l]
        
        
        #join data from catMasks
        cm_ser = pd.read_pickle(cm_fp)['fp'].rename('catMosaic')
        df = df_raw.join(cm_ser)
 
 
        log.info('computing stats on %s' % str(df.shape))
        return df, start
        
    #===========================================================================
    # ERRORS-------
    #===========================================================================
    def run_diffs(self,
                  pick_fp,
                  confusion=False,
                  **kwargs):
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('diffs',  subdir=True,ext='.pkl', **kwargs)
        start = now()
        
        layName_l = ['wse', 'wd']
        #=======================================================================
        # load paths
        #=======================================================================
        df_raw = pd.read_pickle(pick_fp).loc[:, layName_l]
 
        #=======================================================================
        # loop and build diffs for each layer
        #=======================================================================
        res_d = dict()
        for layName, col in df_raw.items():
            
            fp_d = col.to_dict()
        
            #===================================================================
            # run
            #===================================================================
            res_d[layName] = self.get_diffs(fp_d, out_dir=os.path.join(out_dir, layName),
                                            layname=layName, logger=log.getChild(layName),
                                            dry_val={'wse':-9999, 'wd':0.0}[layName],
                                            confusion=confusion)
            
        #=======================================================================
        # wrap
        #=======================================================================
        res_dx = pd.concat(res_d, axis=1, names=['layer']).rename_axis(df_raw.index.name)
        res_dx.to_pickle(ofp)
        
        
        log.info('finished on %s in %.2f secs'%(str(res_dx.shape), (now()-start).total_seconds()))
        
        return ofp
    
    
    def get_diffs(self,fp_d, 
                  write=True,
                  dry_val=-9999,
                  confusion=False,
                   **kwargs):
        """build difference grids for each layer respecting dry masks
        
        
        TODO:
        use xarray and parallelize the delta?
        """
        
        log, tmp_dir, out_dir, _, layname, write = self._func_setup('g', subdir=False, ext='.pkl',write=write, **kwargs)

        #=======================================================================
        # load
        #=======================================================================
        
        log.info('on %i' % len(fp_d))
 
        #===================================================================
        # baseline
        #===================================================================
        base_fp = fp_d[1]
        log.info('from %s' % (os.path.basename(base_fp)))
        
        def read_ds(ds, **kwargs):
            """custom nodata loading"""
            ar_raw = ds.read(1, masked=True, **kwargs) 
            
            #apply the custom mask
            if ds.nodata!=dry_val:
                ar = ma.masked_array(ar_raw.data, mask = ar_raw.data==dry_val, fill_value=-9999)
            else:
                ar = ar_raw
                
            return ar
 
        #===================================================================
        # #load baseline
        #===================================================================
        with rio.open(base_fp, mode='r') as ds: 
            assert ds.res[0] == 1
            base_ar = read_ds(ds) 
            self._base_inherit(ds=ds)
            
        #handle non-native nulls
        """for consistency, treating wd=0 as null for the difference calcs"""

            
        assert base_ar.mask.shape == base_ar.shape        #get the exposure mask
        wets = ~base_ar.mask
        #===================================================================
        # loop on reso
        #===================================================================
        res_d, res_cm_d = dict(), dict()
        for i, (scale, fp) in enumerate(fp_d.items()):
            log.info('    %i/%i scale=%i from %s'%(i+1, len(fp_d), scale, os.path.basename(fp)))
        
            #===============================================================
            # vs. base (no error)
            #===============================================================
            if i==0: 
                res_ar = ma.masked_array(np.full(base_ar.shape, 0), mask=base_ar.mask, fill_value=-9999)
                
                fine_ar = base_ar #for cofusion
 
                
            #===============================================================
            # vs. an  upscale
            #===============================================================
            else:
                #get disagg
                with rio.open(fp, mode='r') as ds:
                    fine_ar = read_ds(ds, out_shape=base_ar.shape, resampling=Resampling.nearest) 
 
                    assert fine_ar.shape==base_ar.shape
                    
                #compute errors
                res_ar = fine_ar - base_ar
 
                
            #===================================================================
            # confusion
            #===================================================================
            if confusion:
                """positive = wet"""
                cm_ar = confusion_matrix(wets.ravel(), ~fine_ar.mask.ravel(),labels=[False, True]).ravel()
     
     
                res_cm_d[scale] =  pd.Series(cm_ar,index = ['TN', 'FP', 'FN', 'TP'])
                
            else:
                res_cm_d[scale] = pd.Series()
            #===============================================================
            # write
            #===============================================================
            assert isinstance(res_ar, ma.MaskedArray)
            assert not np.any(np.isnan(res_ar))
            assert not np.all(res_ar.mask)
        
            if write:
                res_d[scale] = self.write_array(res_ar, 
                                                ofp=os.path.join(out_dir, '%s_diff_%03i.tif'%(layname, scale)), 
                                            logger=log.getChild(f'{scale}'), masked=True)
            else:
                res_d[scale] = np.nan 
            
 
            
        #===================================================================
        # wrap  
        #===================================================================
        
        res_df = pd.concat(res_cm_d, axis=1).T.join(pd.Series(res_d).rename('diff_fp')).rename_axis('subdata', axis=1)
            
 
        #=======================================================================
        # #write
        #=======================================================================
        
        log.debug('finshed on %s'%str(res_df.shape))
        
        
        return res_df
    
 


    def run_diff_stats(self,pick_fp, cm_pick_fp, **kwargs):
        """compute stats from diff rasters filtered by each cat mask
        
        
        Notes
        -----------
        These are all at the base resolution
        
        speedups?
            4 cat masks
            2 layers
            ~6 resolutions
            
            
            distribute to 8 workers: unique cat-mask + layer combinations?
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('diffStats',  subdir=True,ext='.pkl', **kwargs)
        start = now()
        layName_l = ['wse', 'wd']
        
        #=======================================================================
        # load data
        #=======================================================================
        #layers
        df = pd.read_pickle(pick_fp).loc[:, idx[layName_l, 'diff_fp']].droplevel('subdata', axis=1)
        
        #catMasks
        cm_ser = pd.read_pickle(cm_pick_fp)['fp'].rename('catMosaic')
        
        df = df.join(cm_ser)
            
        assert df.isna().sum().sum()==1
        
        #===================================================================
        # loop on each scale
        #===================================================================
        res_lib, meta_d=dict(), dict()
        for i, (scale, ser) in enumerate(df.iterrows()):
            log.info('    %i/%i scale=%i'%(i+1, len(df), scale))
            
            
            #===================================================================
            # setup masks
            #===================================================================
            #the complete mask
            if i==0:
                """just using the ds of the raw wse for shape"""
                with rio.open(ser['wse'], mode='r') as ds:
                    shapeF = ds.shape                    
                    meta_d[scale] = get_pixel_info(ds)
 
                mask_d = {'all':np.full(shapeF, True)} #persists in for loop
                
            #build other masks
            else: 
                with rio.open(ser['catMosaic'], mode='r') as ds:
                    cm_ar = ds.read(1, out_shape=shapeF, resampling=Resampling.nearest, masked=False) #downscale
                    meta_d[scale] = get_pixel_info(ds)
                    
                assert cm_ar.shape==shapeF            
                mask_d.update(self.mosaic_to_masks(cm_ar))
            
            #=======================================================================
            # loop on each layer
            #=======================================================================
            res_d = dict()
            for layName, fp in ser.drop('catMosaic').items():
 
                log.debug('on \'%s\''%layName)

                    
                #===============================================================
                # compute metrics
                #===============================================================
                with rio.open(fp, mode='r') as ds:                    
                    if i==0:
                        rd1 = {'all':{'count':ds.width*ds.height, 'meanErr':0.0, 'meanAbsErr':0.0, 'RMSE':0.0}}

                    else:    
                        ar = ds.read(1, masked=True)    
                        
                        func = lambda x:self._get_diff_stats(x)
                        rd1 = self.get_maskd_func(mask_d, ar, func, log.getChild('%i.%s'%(i, layName)))
                    
 
                #===============================================================
                # wrap layer
                #===============================================================
                res_d[layName] = pd.DataFrame.from_dict(rd1)
            #===================================================================
            # wrap layer loop
            #===================================================================
            res_lib[scale] = pd.concat(res_d, axis=1, names=['layer', 'dsc'])   
            #pd.DataFrame.from_dict(res_d).T.astype({k:np.int32 for k in confusion_l})
        #=======================================================================
        # wrap on layer
        #=======================================================================
        """
        view(res_dx)
        """
        
        res_dx = self._rstats_wrap(res_lib, meta_d, ofp) 
        
        log.info('finished on %s in %.2f secs and wrote to\n    %s'%(str(res_dx.shape), (now()-start).total_seconds(), ofp))
        
        return ofp
    
    def concat_stats(self, fp_d, **kwargs):
        """quick combining and writing of stat pickls"""
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('cstat',  subdir=True,ext='.pkl', **kwargs)
        
        #=======================================================================
        # concat all the picks
        #=======================================================================
        d = dict()
        for base, fp in fp_d.items():
            d[base] = pd.read_pickle(fp)
        
        dx = pd.concat(d, axis=1, names=['base'])
        
        #=======================================================================
        # wrap
        #=======================================================================
        dx.to_pickle(ofp)
        log.info(f'wrote xls {str(dx.shape)} to \n    {ofp}')
        
        if write:
            ofp1 = os.path.join(out_dir, f'{layname}_{len(dx)}_stats.xls')
            with pd.ExcelWriter(ofp1) as writer:       
                dx.to_excel(writer, sheet_name='stats', index=True, header=True)
                
            log.info(f'wrote {str(dx.shape)} to \n    {ofp1}')
                
        return ofp
            
    

            
def get_pixel_info(ds):
    return {'pixelArea':ds.res[0]*ds.res[1], 'pixelLength':ds.res[0]}
    
    
