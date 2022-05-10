'''
Created on Apr. 26, 2022

@author: cefect

small analysis to focus on rasters
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, sys
import qgis.core
from qgis.core import QgsRasterLayer, QgsMapLayerStore
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal


idx = pd.IndexSlice
from hp.exceptions import Error
from hp.pd import get_bx_multiVal
import hp.gdal

from hp.Q import assert_rlay_equal
from agg.hyd.hscripts import Model, StudyArea, view, RasterCalc




class RastRun(Model):
    resCn='resolution'
    saCn='studyArea'
    
    id_params=dict()
    
    def __init__(self,
                 name='rast',
                 data_retrieve_hndls={},
                 lib_dir=None,
                 **kwargs):
        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            #depth rasters
            'drlay_lib':{ #overwrites Model's method
                'compiled':lambda **kwargs:self.load_layer_lib(**kwargs),
                'build':lambda **kwargs: self.build_drlays2(**kwargs),
                },
            'rstats':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_stats(**kwargs),
                },
            'wetArea':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_wetAreas(**kwargs),
                },
            'volume':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_volumes(**kwargs),
                },

            
            #difference rasters
            'difrlay_lib':{  
                'compiled':lambda **kwargs:self.load_layer_lib(**kwargs),
                'build':lambda **kwargs: self.build_difrlays(**kwargs),
                },
            
            'rstatsD':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_stats(**kwargs),
                },
            'difRes':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_difRes(**kwargs), #
                },
            
            #combiners
            'res_dx':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_resdx(**kwargs), #
                },
            
            'res_dx_fp':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_resdxfp(**kwargs), #
                },
                        
            }}
        
        super().__init__( 
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
        
        if lib_dir is None:
            lib_dir = os.path.join(self.work_dir, 'lib', self.name)
        #assert os.path.exists(lib_dir), lib_dir
        self.lib_dir=lib_dir
    #===========================================================================
    # DATA construction-----------
    #===========================================================================
    def runDownsample(self):
        
        #=======================================================================
        # #rasters and stats
        #=======================================================================
        self.retrieve('drlay_lib')
        
        self.retrieve('rstats')
        
        self.retrieve('wetArea')
        
        self.retrieve('volume')
        
        
        

        
        
    def load_layer_lib(self,  # generic retrival for layer librarires
                  fp=None, dkey=None,
                  **kwargs):
        """not the most memory efficient..."""
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('load.%s' % dkey)
        assert dkey in ['drlay_lib', 'difrlay_lib'], dkey
        
        #=======================================================================
        # load the filepaths
        #=======================================================================
        fp_lib = self.load_pick(fp=fp, dkey=dkey)   
        
        #=======================================================================
        # # load layers
        #=======================================================================
        lay_lib = dict()
        cnt = 0
        for k0, d0 in fp_lib.items():
            lay_lib[k0] = dict()
            for k1, fp in d0.items(): #usualy StudyArea
     
                log.info('loading %s.%s from %s' % (k0, k1, fp))
                
                assert isinstance(fp, str), 'got bad type on %s.%s: %s'%(k0, k1, type(fp))
                assert os.path.exists(fp), fp
                ext = os.path.splitext(os.path.basename(fp))[1]
                #===================================================================
                # vectors
                #===================================================================
                if ext in ['.gpkg', '.geojson']:
                
                    lay_lib[k0][k1] = self.vlay_load(fp, logger=log, 
                                                   #set_proj_crs=False, #these usually have different crs's
                                                           **kwargs)
                elif ext in ['.tif']:
                    lay_lib[k0][k1] = self.rlay_load(fp, logger=log, 
                                                   #set_proj_crs=False, #these usually have different crs's
                                                           **kwargs)
                else:
                    raise IOError('unrecognized filetype: %s'%ext)
                cnt+=1
        
        log.info('finished loading %i'%cnt)
        return lay_lib
    
    def build_drlays2(self,
                     
                     #parameters [calc loop]
                     iters=3, #number of downsamples to perform
                     resolution_scale = 2, 
                     base_resolution=None, #resolution of raw data
                     
                     #parameters [get_drlay]. for non base_resolution
                     dsampStage='pre',downSampling='Average',
                     
                     #outputs
                     out_dir=None,
                     
                     dkey='drlay_lib',logger=None,write=None,**kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_drlays')
        if write is None: write=self.write
        assert dkey=='drlay_lib'
        
        #resolutions
        if base_resolution is None:
            from definitions import base_resolution
        
        assert not dsampStage=='none'
        assert not downSampling=='none'
        
        temp_dir = self.temp_dir #collect
        #=======================================================================
        # build iter loop
        #=======================================================================
        #[10, 30, 90]
        resolution_iters = [base_resolution*(resolution_scale)**i for i in range(iters)]
        
        assert max(resolution_iters)<1e5
        #=======================================================================
        # retrive rasters per StudyArea
        #=======================================================================
        """leaving everything on the StudyArea to speed things up"""
        
        #execute
        log.info('constructing %i: %s'%(len(resolution_iters), resolution_iters))
        res_lib = dict()
        cnt=0
        for i, resolution in enumerate(resolution_iters):
            log.info('\n\n%i/%i at %i\n'%(i+1, len(resolution_iters), resolution))
            
            #handle parameters
            if i==0:
                """because get_drlay has expectations for the base
                could also skip the downsampling... but this is mroe consistent"""
                dStage, dSamp='none', 'none'
            else:
                dStage, dSamp=dsampStage, downSampling
                
            #reset temp_dir
            self.temp_dir = os.path.join(temp_dir, 'r%i'%resolution)
            if not os.path.exists(self.temp_dir):os.makedirs(self.temp_dir)
            
            
            #build the depth layer
            try:
                res_lib[resolution] = self.sa_get(meth='get_drlay', logger=log.getChild(str(i)), dkey=dkey, write=False,
                                    resolution=resolution, base_resolution=base_resolution,
                                    dsampStage=dStage, downSampling=dSamp,
                                     **kwargs)
                
                for sa, rlay in res_lib[resolution].items():
                    assert isinstance(rlay, QgsRasterLayer), sa
                
                cnt+=len(res_lib[resolution])
            except Exception as e:
                raise IOError('failed get_drlay on reso=%i w/ \n    %s'%(resolution, e))
 
        self.temp_dir = temp_dir #revert
        log.info('finished building %i'%cnt)
        
 
        #=======================================================================
        # handle layers----
        #=======================================================================
        if write:
            self.store_lay_lib(dkey,  res_lib, out_dir=out_dir, logger=log)
            
        #=======================================================================
        # wrap
        #=======================================================================
        assert_lay_lib(res_lib, msg='%s post'%dkey)
 
            
        return res_lib
    

    
    def build_stats(self, #calc the layer stats 
                    dkey='rstats',
                    logger=None, 
                     **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_stats')
        #assert dkey=='rstats'
        
        #=======================================================================
        # retrieve approriate lib
        #=======================================================================
        if dkey =='rstats':
            lay_lib = self.retrieve('drlay_lib')
        elif dkey=='rstatsD':
            lay_lib = self.retrieve('difrlay_lib')
        else:
            raise IOError(dkey)
        
        #=======================================================================
        # execut ethe function on the stack
        #=======================================================================
        return self.calc_on_layers(
            func=lambda rlay, meta_d={}, **kwargs:self.rlay_getstats(rlay, **kwargs), 
            logger=log, dkey=dkey, lay_lib=lay_lib, **kwargs)
        
    def build_wetAreas(self,
                    dkey='wetArea',
                    logger=None, **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_wetAreas')
        assert dkey=='wetArea'
        
        dx = self.retrieve('rstats')
        
        #=======================================================================
        # define the function
        #=======================================================================
        def func(rlay, logger=None, meta_d={}):
            
            #build a mask layer
            mask_rlay = self.mask_build(rlay, logger=logger, layname='%s_mask'%rlay.name())
            
            #tally all the 1s
            wet_cnt = self.rasterlayerstatistics(mask_rlay)['SUM']
            
            #retrieve stats for this iter
            stats_ser = dx.loc[idx[meta_d['resolution'], meta_d['studyArea']], :]
            
            
            return {dkey:wet_cnt * stats_ser['rasterUnitsPerPixelY']*stats_ser['rasterUnitsPerPixelX']}
 
            
        #=======================================================================
        # execute on stack
        #=======================================================================
        return self.calc_on_layers(func=func, logger=log, dkey=dkey, **kwargs)
        
        """
        view(dx)
        """
    def build_volumes(self,
                    dkey='volume',
                    logger=None, **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_volumes')
        assert dkey=='volume'
        
        dx = self.retrieve('rstats')
        
 
        
        serx = dx['SUM']*dx['rasterUnitsPerPixelX']*dx['rasterUnitsPerPixelY']
        
 
        return serx.rename('volume').to_frame()
        
    
    #===========================================================================
    # DIFF rasters--------
    #===========================================================================
    def runDiffs(self):#run sequence for difference layer calcs
        
        
        self.retrieve('difrlay_lib')
        
        dx = self.retrieve('rstatsD')
        
        self.retrieve('res_dx') #combine all
        
        """
        view(dx.sort_index(level=1))
        """
    

    def store_lay_lib(self, dkey, res_lib,
                      out_dir=None, 
                      logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('store_lay_lib')
        if out_dir is None:
            out_dir = os.path.join(self.wrk_dir, dkey)
        ofp_lib = dict()
        
        #=======================================================================
        # #write each to file
        #=======================================================================
        for resolution, layer_d in res_lib.items():
            ofp_lib[resolution] = self.store_layer_d(layer_d, dkey, logger=log, 
                write_pick=False, #need to write your own
                out_dir=os.path.join(out_dir, 'r%i' % resolution))
        
        #=======================================================================
        # #write the pick
        #=======================================================================
        self.ofp_d[dkey] = self.write_pick(ofp_lib, 
            os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)

    def build_difrlays(self, #generate a set of delta rasters and write to the catalog
                      dkey='difrlay_lib',
                      lay_lib=None,

                   logger=None,
                   out_dir=None,write=None,
 
                      **kwargs):
        """
        revised to match behavior of build_drlays2
        
        NULLS: treating these as zeros for difference calculations
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('gen_rdelta')
        assert dkey=='difrlay_lib'
        saCn=self.saCn
        if write is None: write=self.write
        mstore= QgsMapLayerStore()
        if out_dir is None: out_dir= os.path.join(self.wrk_dir, 'difrlay_lib')
            
        if lay_lib is None: 
            lay_lib = self.retrieve('drlay_lib')
            
        """
        lay_lib.keys()
        """
        
        #=======================================================================
        # loop and execute on each layer
        #=======================================================================
        log.info('on %i'%len(lay_lib))
        
        res_lib=dict()
        base_d = dict()
        first=True
        cnt=0
        for resolution, d0 in lay_lib.items():
            d = dict()
            log.info('\n\nfor resolution=%i building %i delta rasters \n\n'%(resolution, len(d0)))
            for studyArea, rlay in d0.items():
                #setup and precheck
                tagi = '%i.%s'%(resolution, studyArea)
                assert isinstance(rlay, QgsRasterLayer), tagi
                
                #match the layers crs
                self.qproj.setCrs(rlay.crs())
                
                #handle baselines
                if first:
                    base_d[studyArea] = self.get_layer(
                        self.fillnodata(rlay,fval=0, logger=log,
                                        output=os.path.join(self.temp_dir, 
                                                '%s_fnd.tif'%rlay.name())), mstore=mstore)
 
                #execute
                d[studyArea]=self.get_diff_rlay(
                    rlay, base_d[studyArea], #agg - true
                    logger=log.getChild('%i.%s'%(resolution, studyArea)),
                     out_dir = os.path.join(out_dir, studyArea)
                    )
                
                assert isinstance(d[studyArea], QgsRasterLayer)
                cnt+=1
 
            #wrap
            res_lib[resolution] = d
            first=False
            
        #=======================================================================
        # wrap
        #=======================================================================
        #rdx = pd.concat(res_lib, names=['resolution', 'studyArea'])
        
        log.info('finished building %i'%cnt)
        #=======================================================================
        # handle layers----
        #=======================================================================
 
        if write:
            self.store_lay_lib(dkey,  res_lib, out_dir=out_dir, logger=log)
            
        #=======================================================================
        # wrap
        #=======================================================================
        mstore.removeAllMapLayers()
        assert_lay_lib(res_lib, msg='%s post'%dkey)
        return res_lib
 
    #============================================================================
    # COMBINERS------------
    #============================================================================
    def build_resdx(self, #just combing all the results
                     dkey='res_dx',
                     logger=None,write=None,
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_resdx')
        if write is None: write=self.write
        assert dkey=='res_dx'
        resCn = self.resCn 
        saCn = self.saCn
 
        #=======================================================================
        # retrieve and check all
        #=======================================================================
        first = True
        d = dict()
        for dki in ['rstats', 'wetArea','volume', 'rstatsD']:
            dx = self.retrieve(dki)  
            assert np.array_equal(dx.index.names, np.array([self.resCn, self.saCn]))
            
            if first:
                dx_last = dx.copy()
                first = False
            else:      
                assert_index_equal(dx.index, dx_last.index)
                
            d[dki] = dx.sort_index()
        
            
        
        #=======================================================================
        # assemble by type
        #=======================================================================
        #raw values
        raw_dx = d['rstats'].join(d['wetArea']).join(d['volume'])
        
        diff_dx = d['rstatsD']
        
        #difference values
        rdx = pd.concat({'raw':raw_dx, 'diff':diff_dx}, axis=1, names=['rtype', 'stat'])
 
        """
        view(rdx)
        """
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(rdx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return rdx
    
    def build_resdxfp(self, #export everything to the library and write a catalog
                      dkey='res_dx_fp',
                    lib_dir = None, #library directory
                      overwrite=None,
                      compression='med',
                      
                      id_params = {}, #additional parameter values to use as indexers in teh library
                      debug_max_len = None,
                      write=None, logger=None):
        """no cleanup here
        setup for one write per parameterization"""
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey=='res_dx_fp'
        if logger is None: logger=self.logger
        log = logger.getChild(dkey)
        if overwrite is None: overwrite=self.overwrite
        if write is None: write=self.write
        if lib_dir is None:
            lib_dir = self.lib_dir
            
        assert os.path.exists(lib_dir), lib_dir
        resCn=self.resCn
        saCn=self.saCn
        
        #=======================================================================
        # setup filepaths4
        #=======================================================================
        
        rlay_dir = os.path.join(lib_dir, 'rlays', *list(id_params.values()))
 
 
        #=======================================================================
        # re-write raster layers
        #=======================================================================
        """todo: add filesize"""
        ofp_lib = dict()
        for dki in ['drlay_lib', 'difrlay_lib']:
            drlay_lib = self.retrieve(dki)
            #write each to file
            d=dict()
            cnt=0
            for resolution, layer_d in drlay_lib.items():
    
                d[resolution] = self.store_layer_d(layer_d, dki, logger=log,
                                   write_pick=False, #need to write your own
                                   out_dir = os.path.join(rlay_dir,dki, 'r%04i'%resolution),
                                   compression=compression, add_subfolders=False,overwrite=overwrite,                               
                                   )
                
                cnt+=1
                if not debug_max_len is None:
                    if cnt>=debug_max_len:
                        log.warning('cnt>=debug_max_len (%i)... breaking'%debug_max_len)
                        break
                
            dk_clean = dki.replace('_lib', '_fp')
            fp_serx = pd.DataFrame.from_dict(d).stack().swaplevel().rename(dk_clean)
            fp_serx.index.set_names([resCn, saCn], inplace=True)
            ofp_lib[dk_clean] = fp_serx
            
        #collect
        fp_dx = pd.concat(ofp_lib, axis=1)
        
        

        #=======================================================================
        # build catalog
        #=======================================================================
        rdx_raw = self.retrieve('res_dx')
        
        #=======================================================================
        # #add filepaths
        #=======================================================================
 
        #promote columns on filepaths to match
        cdf = pd.Series(fp_dx.columns, name='fp').to_frame()
        cdf['rtype'] = rdx_raw.columns.unique('rtype')
        cdf['fp']='fp' #drop redundant info.. easier for post analyssis
        fp_dx.columns = pd.MultiIndex.from_frame(cdf).swaplevel()
        
        #join
        rdx = rdx_raw.join(fp_dx).sort_index()
        
        assert rdx.notna().any().any()
        
        #add additional id params
        #add singgle value levels from a dictionary
        mdex_df = rdx.index.to_frame().reset_index(drop=True)
        for k,v in id_params.items():
            mdex_df[k] = v
            
            #remove from cols
            if k in rdx.columns.get_level_values(1):
                """TODO: check the values are the same"""
                rdx = rdx.drop(k, level=1, axis=1)
            
        rdx.index = pd.MultiIndex.from_frame(mdex_df)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(rdx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        
        return rdx

    def write_lib(self,
                  catalog_fp=None,
                  lib_dir = None, #library directory
                  rdx = None,
                  overwrite=None, logger=None, **kwargs):
        
        if overwrite is None: overwrite=self.overwrite
        if logger is None: logger=self.logger
        log=logger.getChild('write_lib')
        
        
        if lib_dir is None:
            lib_dir = self.lib_dir
            
        if catalog_fp is None: 
            catalog_fp = os.path.join(lib_dir, '%s_run_index.csv'%self.name)
            
        if rdx is None:
            rdx=self.retrieve('res_dx_fp', lib_dir=lib_dir, **kwargs)
        #=======================================================================
        # write catalog
        #=======================================================================
        miss_l = set(rdx.index.names).symmetric_difference(Catalog.keys)
        assert len(miss_l)==0, 'key mistmatch with catalog worker'
        
        with Catalog(catalog_fp=catalog_fp, overwrite=overwrite, logger=log) as cat:
            for rkeys, row in rdx.iterrows():
                keys_d = dict(zip(rdx.index.names, rkeys))
                cat.add_entry(row, keys_d, logger=log.getChild(str(rkeys)))
        
        
        """
        rdx.index.names.to_list()
        view(rdx)
        """
        log.info('finished')
        return catalog_fp
    


    #===========================================================================
    # HELPERS-------
    #===========================================================================
    def calc_on_layers(self,
                       #data
                       lay_lib=None,
                       
                       #parameters
                       func=lambda rlay, **kwargs:{},
                       
                       #writing
                       write=None, dkey=None,
                       
                       logger=None, **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('calcLayers')
        
        if lay_lib is None: lay_lib = self.retrieve('drlay_lib')
        
        if write is None: write=self.write
        
        #=======================================================================
        # loop and execute on each layer
        #=======================================================================
        log.info('on %i'%len(lay_lib))
        
        res_d=dict()
        for resolution, d0 in lay_lib.items():
            d = dict()
            for studyArea, rlay in d0.items():
                #setup and precheck
                tagi = '%i.%s'%(resolution, studyArea)
                if not isinstance(rlay, QgsRasterLayer):
                    raise Error('got bad type on %s: %s'%(
                        tagi, type(rlay)))
                
                #match the layers crs
                self.qproj.setCrs(rlay.crs())
                
                #execute
                res = func(rlay, logger=log.getChild(tagi),meta_d={'studyArea':studyArea, 'resolution':resolution}, **kwargs)
                
                #post
                assert isinstance(res, dict)                
                d[studyArea]=res
                
            #wrap
            res_d[resolution] = pd.DataFrame(d).T
            
        #=======================================================================
        # wrap
        #=======================================================================
        rdx = pd.concat(res_d, names=['resolution', 'studyArea'])
        
        assert isinstance(rdx, pd.DataFrame)
        
        log.info('finished on %i'%len(rdx))
        
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        
        return rdx
    
    def get_diff_rlay(self, #top minus bottom (agg - base)
                      top_rlay, bot_rlay, 
                      
                      base_resolution=None,
                      out_dir=None,
                      logger=None,
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_diff_rlay')
        
        temp_dir = os.path.join(self.temp_dir, 'get_diff_rlay')
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        if out_dir is None: out_dir = os.path.join(self.temp_dir, 'difrlay_lib')
        start = datetime.datetime.now()
        mstore = QgsMapLayerStore()
        
        if base_resolution is None:
            from definitions import base_resolution
        
        
        extents = self.layerextent(bot_rlay,precision=0.0, ).extent()
 
        """pretty slow"""
        assert self.rlay_get_resolution(bot_rlay)==float(base_resolution)
        
        #=======================================================================
        # fill nulls
        #=======================================================================
        assert hp.gdal.getNoDataCount(bot_rlay.source())==0
        
        topr1_fp = self.fillnodata(top_rlay, fval=0, logger=log,
                           output=os.path.join(temp_dir, '%s_fnd.tif'%top_rlay.name()))
 
        #=======================================================================
        # warop top to match
        #=======================================================================
        tres = self.rlay_get_resolution(top_rlay)
        if tres > float(base_resolution):
            log.info('warpreproject w/ resolution=%i to %s'%(base_resolution, extents))
            topr2_fp = self.warpreproject(topr1_fp, compression='none', extents=extents, logger=log,
                                            resolution=base_resolution,
                                            output=os.path.join(
                                                temp_dir, 'preWarp_%000i_%s'%(int(base_resolution), os.path.basename(top_rlay.source()))
                                                ))
        elif tres < float(base_resolution):
            raise IOError(tres)
 
        else:
            topr2_fp = topr1_fp
 
        #=======================================================================
        # subtract
        #=======================================================================
        
        log.debug('building RasterCalc')
        with RasterCalc(topr2_fp, name='diff', session=self, logger=log,out_dir=out_dir,
                        ) as wrkr:
 
            entries_d = {k:wrkr._rCalcEntry(v) for k,v in {
                'top':wrkr.ref_lay, 'bottom':bot_rlay}.items()}
            
            assert_rlay_equal(entries_d['top'].raster, entries_d['bottom'].raster)
            
            formula = '%s - %s'%(entries_d['top'].ref, entries_d['bottom'].ref)
            
            #===================================================================
            # execute subtraction
            #===================================================================
            log.info('executing %s'%formula)
            diff_fp1 = wrkr.rcalc(formula, layname='diff_%s'%os.path.basename(topr1_fp).replace('.tif', ''))
            
        #=======================================================================
        # null check
        #=======================================================================
        null_cnt = hp.gdal.getNoDataCount(diff_fp1)
        if not null_cnt==0:
            basename, ext = os.path.splitext(diff_fp1)
            """not sure why this is happenning for some layers"""
            log.warning('got %i nulls on diff for %s...filling'%(null_cnt, top_rlay.name()))
            diff_fp2 = self.fillnodata(diff_fp1, fval=0.0, logger=log,
                                       output=basename+'_fnd.tif')
        else:
            diff_fp2 = diff_fp1
        
        
 
        mstore.removeAllMapLayers()
        return self.rlay_load(diff_fp2, logger=log)
 
class Catalog(object): #handling the simulation index and library
    df=None
    keys = ['resolution', 'studyArea', 'downSampling', 'dsampStage', 'severity']
    cols = ['rtype', 'stat']
 

    
    def __init__(self, 
                 catalog_fp='fp', 
                 logger=None,
                 overwrite=True,
                 ):
        
        if logger is None:
            import logging
            logger = logging.getLogger()
            

        

        
        #=======================================================================
        # attachments
        #=======================================================================
        self.logger = logger.getChild('cat')
        self.overwrite=overwrite
        self.catalog_fp = catalog_fp
        
        
        #mandatory keys
        self.cat_colns = ['cell_cnt']
        
        #=======================================================================
        # load existing
        #=======================================================================
        if os.path.exists(catalog_fp):
            self.df = pd.read_csv(catalog_fp, 
                                  index_col=list(range(len(self.keys))),
                                  header = list(range(len(self.cols))),
                                  )
            self.check(df=self.df.copy())
            self.df_raw = self.df.copy() #for checking
            
        else:
            self.df_raw=pd.DataFrame()
        
    def clean(self):
        raise Error('check consitency between index and library contents')
    
    def check(self,
              df=None,
              ):
        #=======================================================================
        # defai;lts
        #=======================================================================
        log = self.logger.getChild('check')
        if df is None: df = self.df.copy()
        log.debug('on %s'%str(df.shape))
        
        #check columns

        miss_l = set(self.cols).difference(df.columns.names)
        assert len(miss_l)==0, miss_l
        
        assert isinstance(df.columns, pd.MultiIndex)
        
 
        #check index
        #=======================================================================
        # assert df[self.idn].is_unique
        # assert 'int' in df[self.idn].dtype.name
        #=======================================================================
        assert isinstance(df.index, pd.MultiIndex)
        assert np.array_equal(np.array(df.index.names), np.array(self.keys))
        
        #check filepaths
        errs_d = dict()
        
        bx_col = df.columns.get_level_values(1).str.endswith('fp')
        assert bx_col.any()
        
        for coln, cserx in df.loc[:, bx_col].items():
 
            
            for id, path in cserx.items():
                if pd.isnull(path):
                    log.warning('got null filepath on %s'%str(id))
                    continue
                if not os.path.exists(path):
                    errs_d['%s_%s'%(coln, id)] = path
                    
        if len(errs_d)>0:
            log.error(errs_d)
            raise Error('got %i/%i bad filepaths'%(len(errs_d), len(df)))
 
 
        
    
    def get(self):
        assert os.path.exists(self.catalog_fp), self.catalog_fp
        self.check()
        return self.df.copy().sort_index()
    
    
    def remove(self, keys_d,
               logger=None): #remove an entry
        #=======================================================================
        # defaults 
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('remove')
        df_raw = self.df.copy()
        
        #identify this row
        bx = pd.Series(True, index=df_raw.index)
        for k,v in keys_d.items():
            bx = np.logical_and(bx,
                                df_raw.index.get_level_values(k)==v,
                                )
            
        assert bx.sum()==1
        
        #remove the raster
        """no... this should be handled by the library writer
        rlay_fp = df_raw.loc[bx, 'rlay_fp'].values[0]
        
        os.remove(rlay_fp)"""
            
        
        #remove the row
        self.df = df_raw.loc[~bx, :]
        
        log.info('removed %s'%(keys_d))
        
 

    def add_entry(self,
                  serx,
                  keys_d={},
                  logger=None,
                  ):
        """
        cat_d.keys()
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('add_entry')
        cat_df = self.df
        keys = self.keys.copy()

        log.info('w/ %i'%len(serx))
        #check mandatory columns are there
        miss_l = set(self.cat_colns).difference(serx.index.get_level_values(1))
        assert len(miss_l)==0, 'got %i unrecognized keys: %s'%(len(miss_l), miss_l)
        
        for k in keys: 
            assert k in keys_d
        
        #=======================================================================
        # prepare new 
        #=======================================================================
        
        #new_df = pd.Series(cat_d).rename(keys_d.values()).to_frame().T
 
        new_df = serx.to_frame().T
        new_df.index = pd.MultiIndex.from_tuples([keys_d.values()], names=keys_d.keys())
        """
        view(new_df)
        """
        #=======================================================================
        # append
        #=======================================================================
        if cat_df is None:
            #convet the series (with tuple name) to a row of a multindex df
            cat_df=new_df
        else:
            #check if present
            
            cdf = cat_df.index.to_frame().reset_index(drop=True)
            ndf = new_df.index.to_frame().reset_index(drop=True)
 
            bxdf = cdf.apply(lambda s:ndf.eq(s).iloc[0,:], axis=1)
            bx = bxdf.all(axis=1)
 
            
            
            if bx.any():
                
                #===============================================================
                # remomve an existing entry
                #===============================================================
                assert bx.sum()==1
                assert self.overwrite
                bx.index = cat_df.index
                self.remove(dict(zip(keys, cat_df.loc[bx, :].iloc[0,:].name)))
                
                cat_df = self.df.copy()
                
            
            #===================================================================
            # append
            #===================================================================
            cat_df = cat_df.append(new_df,  verify_integrity=True)
            
        #=======================================================================
        # wrap
        #=======================================================================
        self.df = cat_df
 
        
        log.info('added %s'%len(keys_d))
                                
        

 
        
        
    def __enter__(self):
        return self
    def __exit__(self, *args, **kwargs):
        if self.df is None: 
            return
        #=======================================================================
        # write if there was a change
        #=======================================================================
        if not np.array_equal(self.df, self.df_raw):
            log = self.logger.getChild('__exit__')
            df = self.df
            
            #===================================================================
            # delete the old for empties
            #===================================================================
 
            if len(df)==0:
                try:
                    os.remove(self.catalog_fp)
                    log.warning('got empty catalog... deleteing file')
                except Exception as e:
                    raise Error(e)
            else:
                #===============================================================
                # write the new
                #===============================================================
                """should already be consolidated... just overwrite"""
 
                df.to_csv(self.catalog_fp, mode='w', header = True, index=True)
                self.logger.info('wrote %s to %s'%(str(df.shape), self.catalog_fp))
        
#===============================================================================
# funcs
#===============================================================================
def assert_lay_lib(lib_d, msg=''):
    if __debug__:
        assert isinstance(lib_d, dict)
        for k0,d0 in lib_d.items():
            if not isinstance(d0, dict):
                raise AssertionError('bad subtype on %s: %s\n'%(
                    k0, type(d0))+msg)
            
            for k1, lay in d0.items():
                if not isinstance(lay, QgsRasterLayer):
                    raise AssertionError('bad type on %s.%s: %s\n'%(
                        k0,k1, type(lay))+msg)


 