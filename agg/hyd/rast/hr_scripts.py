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

np.random.seed(100)
idx = pd.IndexSlice
from hp.exceptions import Error
from hp.pd import get_bx_multiVal
import hp.gdal
from agg.hyd.hscripts import Model, StudyArea, view, RasterCalc

start = datetime.datetime.now()
print('start at %s' % start)


class RastRun(Model):
    resCn='resolution'
    saCn='studyArea'
    
    id_params=dict()
    
    def __init__(self,
                 name='rast',
                 data_retrieve_hndls={},
                 **kwargs):
        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            'drlay_lib':{ #overwrites Model's method
                'compiled':lambda **kwargs:self.load_layer_lib(**kwargs),
                'build':lambda **kwargs: self.build_drlays2(**kwargs),
                },
            'rstats':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_stats(**kwargs),
                },
            'wetAreas':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_wetAreas(**kwargs),
                },
            'res_dx':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_resdx(**kwargs), #
                },
            
            #difference layers
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
                        
            }}
        
        super().__init__( 
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
    #===========================================================================
    # DATA construction-----------
    #===========================================================================
    def runDownsample(self):
        
        #=======================================================================
        # #rasters and stats
        #=======================================================================
        self.retrieve('drlay_lib')
        
        self.retrieve('rstats')
        
        self.retrieve('wetAreas')
        
        
        

        
        
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
                     dsampStage='wse',downSampling='Average',
                     
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
                
                cnt+=len(res_lib[resolution])
            except Exception as e:
                raise IOError('failed on %i w/ \n    %s'%(resolution, e))
 
        self.temp_dir = temp_dir #revert
        log.info('finished building %i'%cnt)
        #=======================================================================
        # handle layers----
        #=======================================================================
 
        if write:
            if out_dir is None: out_dir = os.path.join(self.wrk_dir, dkey)
            ofp_lib = dict()
            
            #write each to file
            for resolution, layer_d in res_lib.items():
                ofp_lib[resolution] = self.store_layer_d(layer_d, dkey, logger=log,
                                   write_pick=False, #need to write your own
                                   out_dir = os.path.join(out_dir, 'r%i'%resolution)
                                   )
                
            #write the pick
            self.ofp_d[dkey] = self.write_pick(ofp_lib,
                os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
            
        #=======================================================================
        # wrap
        #=======================================================================
 
            
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
                    dkey='wetAreas',
                    logger=None, **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_wetAreas')
        assert dkey=='wetAreas'
        
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
        return self.calc_on_layers(
            func=func, 
            logger=log, dkey=dkey, **kwargs)
        
        """
        view(dx)
        """
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
        # retrieve from hr_scripts
        #=======================================================================
        first = True
        d = dict()
        for dki in ['rstats', 'wetAreas', 'rstatsD']:
            dx = self.retrieve(dki)  
            assert np.array_equal(dx.index.names, np.array([self.resCn, self.saCn]))
            
            if first:
                dx_last = dx.copy()
                first = False
            else:      
                assert_index_equal(dx.index, dx_last.index)
                
            d[dki] = dx.sort_index()
        
            
        
        #=======================================================================
        # join
        #=======================================================================
        #raw values
        rdx1 = d['rstats'].join(d['wetAreas'])
        
        #difference values
        rdx = pd.concat({'raw':rdx1, 'diff':d['rstatsD']}, axis=1, names=['rtype', 'stat'])
 
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
    
    def build_difrlays(self, #generate a set of delta rasters and write to the catalog
                      dkey='difrlay_lib',
                      lay_lib=None,

                   logger=None,
                   out_dir=None,
 
                      **kwargs):
        """
        NOTE: this always writes to file
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('gen_rdelta')
        assert dkey=='difrlay_lib'
        saCn=self.saCn
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
                    base_d[studyArea] = self.rlay_mcopy(rlay, mstore=mstore)
                    
 
                
                #execute
                d[studyArea]=self.get_diff_rlay(
                    rlay, base_d[studyArea], #agg - true
                    logger=log.getChild('%i.%s'%(resolution, studyArea)),
                     out_dir = os.path.join(out_dir, studyArea)
                    )
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
        self.ofp_d[dkey] = self.write_pick(res_lib,
            os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
            
        #=======================================================================
        # wrap
        #=======================================================================
 
        return res_lib
        
   
 
        mstore.removeAllMapLayers()
        
        
 

        
    
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
                assert isinstance(rlay, QgsRasterLayer), tagi
                
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
        if out_dir is None: out_dir = os.path.join(self.wrk_dir, 'difrlay_lib')
        start = datetime.datetime.now()
        mstore = QgsMapLayerStore()
        
        if base_resolution is None:
            from definitions import base_resolution
        
        
        extents = self.layerextent(bot_rlay, 
                                   precision=0.0, #adding this buffer causes some problems with the tests
                                   ).extent()
 
        """pretty slow"""
        assert self.rlay_get_resolution(bot_rlay)==float(base_resolution)
 
        #=======================================================================
        # warop top to match
        #=======================================================================
        if not self.rlay_get_resolution(top_rlay)==float(base_resolution):
            log.info('warpreproject w/ resolution=%i to %s'%(base_resolution, extents))
            topr1_fp = self.warpreproject(top_rlay, compression='none', extents=extents, logger=log,
                                            resolution=base_resolution,
                                            output=os.path.join(
                                                temp_dir, 'preWarp_%000i_%s'%(int(base_resolution), os.path.basename(top_rlay.source()))
                                                ))
            
            topr1_fp = self.fillnodata(topr1_fp, output=os.path.join(
                                                temp_dir, 'preWarp_fnd_%000i_%s'%(int(base_resolution), os.path.basename(top_rlay.source()))
                                                ))
            #===================================================================
            # mstore.addMapLayer(topr1_rlay)
            # topr1_fp = topr1_rlay.source()
            #===================================================================
            
            
        else:
            topr1_fp = top_rlay.source()
            
        #=======================================================================
        # check
        #=======================================================================
        """a bit slow"""
        assert hp.gdal.getNoDataCount(topr1_fp)==0
        assert hp.gdal.getNoDataCount(bot_rlay.source())==0
            
        #=======================================================================
        # subtract
        #=======================================================================
 
        
        log.debug('building RasterCalc')
        with RasterCalc(topr1_fp, name='diff', session=self, logger=log,out_dir=out_dir,) as wrkr:
            
            top_rlay = wrkr.ref_lay #loaded during init
            #bot_rlay = wrkr.load(bot_rlay)
            
            #===================================================================
            # setup
            #===================================================================
            
            entries_d = {k:wrkr._rCalcEntry(v) for k,v in {'top':top_rlay, 'bottom':bot_rlay}.items()}
            formula = '%s - %s'%(entries_d['top'].ref, entries_d['bottom'].ref)
            
            #===================================================================
            # execute subtraction
            #===================================================================
            log.info('executing %s'%formula)
            diff_fp = wrkr.rcalc(formula, layname='diff_%s'%os.path.basename(topr1_fp))
            
        #=======================================================================
        # wrap
        #=======================================================================
        mstore.removeAllMapLayers()
        assert hp.gdal.getNoDataCount(diff_fp)==0
        
        return diff_fp
 
        
                
    def write_lib(self, #export everything to the library and write a catalog
                    lib_dir = None, #library directory
                      overwrite=None,
                      compression='med',
                      catalog_fp=None,
                      id_params = {}, #additional parameter values to use as indexers in teh library
                      debug_max_len = None,
                      ):
        """no cleanup here
        setup for one write per parameterization"""
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('write_lib')
        if overwrite is None: overwrite=self.overwrite
        if lib_dir is None:
            lib_dir = os.path.join(self.work_dir, 'lib', self.name)
            
        assert os.path.exists(lib_dir), lib_dir
        resCn=self.resCn
        saCn=self.saCn
        
        #=======================================================================
        # setup filepaths4
        #=======================================================================
        if catalog_fp is None: catalog_fp = os.path.join(lib_dir, '%s_run_index.csv'%self.name)
        rlay_dir = os.path.join(lib_dir, 'rlays', *list(id_params.values()))
 
        #=======================================================================
        # retrieve------
        #=======================================================================
 
        
        #=======================================================================
        # re-write raster layers
        #=======================================================================
        """todo: add filesize"""
        ofp_lib = dict()
        for dkey in ['drlay_lib', 'difrlay_lib']:
            drlay_lib = self.retrieve(dkey)
            #write each to file
            d=dict()
            cnt=0
            for resolution, layer_d in drlay_lib.items():
    
                d[resolution] = self.store_layer_d(layer_d, dkey, logger=log,
                                   write_pick=False, #need to write your own
                                   out_dir = os.path.join(rlay_dir,dkey, 'r%04i'%resolution),
                                   compression=compression, add_subfolders=False,overwrite=overwrite,                               
                                   )
                
                cnt+=1
                if not debug_max_len is None:
                    if cnt>=debug_max_len:
                        log.warning('cnt>=debug_max_len (%i)... breaking'%debug_max_len)
                        break
                
            dk_clean = dkey.replace('_lib', '_fp')
            fp_serx = pd.DataFrame.from_dict(d).stack().swaplevel().rename(dk_clean)
            fp_serx.index.set_names([resCn, saCn], inplace=True)
            ofp_lib[dk_clean] = fp_serx
            
        #collect
        fp_dx = pd.concat(ofp_lib, axis=1)
        
        

        #=======================================================================
        # build catalog
        #=======================================================================
        rdx_raw = self.retrieve('res_dx')
        
        #add filepaths
 
        #assert_index_equal(fp_dx.index.sort_values(), rdx_raw.index.sort_values())
        
        
        #promote columns on filepaths to match
        cdf = pd.Series(fp_dx.columns, name='fp').to_frame()
        cdf['rtype'] = rdx_raw.columns.unique('rtype')
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
        
        bx_col = df.columns.get_level_values(1).str.endswith('_fp')
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
            
            bx = cat_df.index.to_frame().reset_index(drop=True).eq(
                new_df.index.to_frame().reset_index(drop=True), 
                axis=1).all(axis=1)
            
            
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
# class StudyArea2(RastRun, StudyArea):
#     pass
#===============================================================================

def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='tag',
        name='hrast0',
        overwrite=True,
        trim=False,
        
        
        
        #=======================================================================
        # write control
        #=======================================================================
        write=True,
        exit_summary=True,
        write_lib=True, #enter the results into the library
 
        compression='med',
        #=======================================================================
        # #data
        #=======================================================================
        studyArea_l = None, #convenience filtering of proj_lib
        proj_lib = None,
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

 
        #=======================================================================
        # #parameters
        #=======================================================================
        iters=3, #resolution iterations
        #raster downSampling and selection  (StudyArea.get_raster())
        dsampStage='wse', downSampling='Average', severity = 'hi', 
        #resolution=5, this is what we iterate on
        
 
        
        #=======================================================================
        # debug
        #=======================================================================
        debug_max_len=None,
 
        
 
        **kwargs):
    print('START run w/ %s.%s and '%(name, tag))
 
    #===========================================================================
    # study area filtering
    #===========================================================================
    if proj_lib is None:
        from definitions import proj_lib
    
    if not studyArea_l is None:
        print('filtering studyarea to %i: %s'%(len(studyArea_l), studyArea_l))
        miss_l = set(studyArea_l).difference(proj_lib.keys())
        assert len(miss_l)==0, 'passed %i studyAreas not in proj_lib: %s'%(len(miss_l), miss_l)
        proj_lib = {k:v for k,v in proj_lib.items() if k in studyArea_l}
    #===========================================================================
    # execute
    #===========================================================================
    with RastRun(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                     write=write,exit_summary=exit_summary,prec=prec,
                 bk_lib = {
 
                     
                     'drlay_lib':dict( severity=severity, downSampling=downSampling, dsampStage=dsampStage, iters=iters),
 
                                          
                     },
                 **kwargs) as ses:
        
        ses.runDownsample()
        
        ses.runDiffs()
        
        if write_lib:
            ses.write_lib(compression=compression, id_params=dict(downSampling=downSampling, dsampStage=dsampStage, severity=severity), debug_max_len=debug_max_len)
        
 
        
 
        
    print('\nfinished %s'%tag)
    
    return 

def dev():
    return run(
        trim=True, compression='none',name='dev',
        tag='dev',
        iters=2,
        compiled_fp_d={
            'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\rast\dev\20220427\working\drlay_lib_rast_dev_0427.pickle',
            }
        )

 

def r2():
    return run(tag='wse', name='hrast1',iters=10,
               dsampStage='wse', 
               downSampling='Average',
               compiled_fp_d = {
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\drlay_lib_hrast1_wse_0427.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\rstats_basic_hrast1_wse_0427.pickle',
        'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\wetAreas_hrast1_wse_0427.pickle',
 
        
        
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\difrlay_lib_hrast1_wse_0505.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\rstatsD_hrast1_wse_0505.pickle',
        
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\res_dx_hrast1_wse_0505.pickle',
        },
               write_lib=True,
               )
               
def r3_wse():
    return run(tag='wse', name='hrast2',iters=10,
               dsampStage='wse', 
               downSampling='Average',
               compiled_fp_d = {
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\drlay_lib_hrast1_wse_0427.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\rstats_basic_hrast1_wse_0427.pickle',
        'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220427\working\wetAreas_hrast1_wse_0427.pickle',
 
        
        
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\difrlay_lib_hrast1_wse_0505.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\rstatsD_hrast1_wse_0505.pickle',
        
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hrast1\wse\20220505\working\res_dx_hrast1_wse_0505.pickle',
        },
               write_lib=True,
               #debug_max_len=2,
               )
    
def r3_depth():
    return run(tag='depth', name='hrast2',iters=10,
               dsampStage='depth', 
               downSampling='Average',
               compiled_fp_d = {
                   'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\drlay_lib_hrast2_depth_0505.pickle',
                   'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\difrlay_lib_hrast2_depth_0505.pickle',
                   'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\rstats_hrast2_depth_0505.pickle',
                    'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\wetAreas_hrast2_depth_0505.pickle',
                    'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\rstatsD_hrast2_depth_0505.pickle',
                    'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hrast2\depth\20220505\working\res_dx_hrast2_depth_0505.pickle',
                                },
 
               )
if __name__ == "__main__": 
    
    r3_depth()

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
 