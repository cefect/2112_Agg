'''
Created on Apr. 26, 2022

@author: cefect

analysis to focus on rasters
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
from pathlib import Path

idx = pd.IndexSlice
from hp.exceptions import Error
from hp.pd import get_bx_multiVal
import hp.gdal

from hp.Q import assert_rlay_equal, QgsMapLayer, view
from hp.basic import set_info
from agg.hyd.hscripts import  RasterCalc

from agg.hydR.hydR_coms import RRcoms, Catalog, assert_lay_lib
    


class RastRun(RRcoms):
    
    phase_l=['depth', 'diff']
    index_col = list(range(5))
    
    def __init__(self,
                 name='rast',
                 phase_l=['depth'],
                 data_retrieve_hndls={},
                 rcol_l=None, 
                 pick_index_map={},
                 **kwargs):
        
        
        if rcol_l is None:
            rcol_l=[self.saCn, self.resCn]
        self.rcol_l=rcol_l
        
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
            'wetStats':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_wetStats(**kwargs),
                },
 
            'gwArea':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_gwArea(**kwargs),
                },
            
            'noData_cnt':{  #note, rstats will also have this
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                #'build':lambda **kwargs: self.build_stats(**kwargs),
                },
 
            'noData_pct':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_noDataPct(**kwargs),
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
            'rmseD':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_rmseD(**kwargs),
                },
 
            
            #combiners
            'res_dx':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_resdx(**kwargs), #
                },
            
            'layxport':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_layxport(**kwargs), #
                },
            
 
                        
            }}
        
        pick_index_map.update({
            'drlay_lib':(self.resCn, self.saCn),
            })
        self.pick_index_map=pick_index_map
        
        super().__init__( 
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
        self.phase_l=phase_l
        
    #===========================================================================
    # COMPILEERS----
    #===========================================================================
    def compileFromCat(self, #construct pickle from the catalog and add to compiled
                       catalog_fp='',
                       dkey_l = ['drlay_lib'], #dkeys to laod
                       
                       id_params={}, #index values identifying this run
                       
                       logger=None,
                       pick_index_map=None,
                       ):
        """
        because we generally execute a group of parameterizations (id_params) as 1 run (w/ a batch script)
            then compile together in the catalog for analysis
            
        loading straight from the catalog is nice if we want to add one calc to the catalog set
        
        our framework is still constrained to only execute 1 parameterization per call
            add a wrapping for loop to execute on the whole catalog
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        if pick_index_map is None: pick_index_map=self.pick_index_map
        log=logger.getChild('compileFromCat')
        
        for dkey in dkey_l:
            assert dkey in pick_index_map
            assert not dkey in self.compiled_fp_d, dkey
        #=======================================================================
        # load the catalog
        #=======================================================================
        
        with Catalog(catalog_fp=catalog_fp, logger=logger, overwrite=False,
                       index_col=self.index_col ) as cat:
            
            for dkey in dkey_l:
                log.info('\n\n on %s\n\n'%dkey)
                
                #pull the filepaths from the catalog
                fp_d = cat.get_dkey_fp(dkey=dkey, pick_indexers=pick_index_map[dkey], id_params=id_params)
                
                #save as a pickel
                """writing to temp as we never store these"""
                self.compiled_fp_d[dkey] = self.write_pick(fp_d, 
                                    os.path.join(self.temp_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
                
        log.info('finished on %i'%len(dkey_l))
        
        return
                
            
        
        

    #===========================================================================
    # RASTER DATA construction-----------
    #===========================================================================
    def runDownsample(self):
        
        #=======================================================================
        # #rasters and stats
        #=======================================================================
        self.retrieve('drlay_lib')
        
        self.retrieve('rstats')
        
        self.retrieve('wetStats')
        
 
        
        self.retrieve('gwArea')
        
 
        
        self.retrieve('noData_pct')
        
        
        

        
        
    def load_layer_lib(self,  # generic retrival for layer librarires
                  fp=None, dkey=None,
                  **kwargs):
        """not the most memory efficient..."""
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('load.%s' % dkey)
        assert dkey in ['drlay_lib', 'difrlay_lib', 'finv_agg_lib', 'finv_sg_lib'], dkey
        
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
                     proj_lib=None,
                     
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
        # clean proj_lib
        #=======================================================================
        if proj_lib is None:
            proj_lib = copy.deepcopy(self.proj_lib)
        for sa, d in proj_lib.items():
            if 'finv_fp' in d:
                del d['finv_fp']
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
        nd_res_lib=dict()
        cnt=0
        for i, resolution in enumerate(resolution_iters):
            log.info('\n\n%i/%i at %i\n'%(i+1, len(resolution_iters), resolution))
            
            #===================================================================
            # #handle parameters
            #===================================================================
            if i==0:
                """because get_drlay has expectations for the base
                could also skip the downsampling... but this is mroe consistent"""
                dStage, dSamp='none', 'none'
            else:
                dStage, dSamp=dsampStage, downSampling
                
            #reset temp_dir
            self.temp_dir = os.path.join(temp_dir, 'r%i'%resolution)
            if not os.path.exists(self.temp_dir):os.makedirs(self.temp_dir)
            
            
            #===================================================================
            # #build the depth layer
            #===================================================================
            try:
                d = self.sa_get(meth='get_drlay', logger=log.getChild(str(i)), 
                                                  dkey=dkey, write=False,
                                    resolution=resolution, base_resolution=base_resolution,
                                    dsampStage=dStage, downSampling=dSamp,proj_lib=proj_lib,
                                     **kwargs)
                
                #extract
                res_lib[resolution] = {k:d0.pop('rlay') for k, d0 in d.items()}
                for sa, rlay in res_lib[resolution].items():
                    assert isinstance(rlay, QgsRasterLayer), sa
                
                cnt+=len(res_lib[resolution])
                
                #nodata counts
                nd_res_lib[resolution] = pd.DataFrame.from_dict(d)
                
            except Exception as e:
                raise IOError('failed get_drlay on reso=%i w/ \n    %s'%(resolution, e))
 
        self.temp_dir = temp_dir #revert
        log.info('finished building %i'%cnt)
        
 
        #=======================================================================
        # handle layers----
        #=======================================================================
        if write:
            self.store_lay_lib(  res_lib, dkey,out_dir=out_dir, logger=log)
            
        #=======================================================================
        # handle meta
        #=======================================================================
        if write:
            rdx = pd.concat(nd_res_lib).T.stack(level=0).swaplevel().sort_index(sort_remaining=True)
            rdx.index.set_names([self.resCn, self.saCn], inplace=True)
            
            #rename to not conflict
            assert len(rdx.columns)==1
            rdx.columns = ['noData_cnt2']
            
            assert np.array_equal(rdx.index.names, np.array([self.resCn, self.saCn]))
            
            dkey1 = 'noData_cnt'
            self.data_d[dkey1] = rdx.copy()
            self.ofp_d[dkey1] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey1, self.longname)),
                                   logger=log)
            
            
        #=======================================================================
        # wrap-----
        #=======================================================================
 
        assert_lay_lib(res_lib, msg='%s post'%dkey)
 
            
        return res_lib
    
 
    def build_stats(self, #calc the layer stats 
                    dkey='rstats',
                    logger=None, 
                    lay_lib=None,
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
        if lay_lib is None:
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
        
    def build_wetStats(self, #calc wetted stats
                    dkey='wetStats',
                    logger=None, **kwargs):
 
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_wetStats')
        assert dkey=='wetStats'
        
        dx = self.retrieve('rstats')
        """
        dx.index
        """
        
        #=======================================================================
        # define the function
        #=======================================================================
        def func(rlay, logger=None, meta_d={}):
            
            
            assert hp.gdal.getNoDataCount(rlay.source())==0
            
            #build a mask layer
            mask_rlay = self.mask_build(rlay, logger=logger, layname='%s_mask'%rlay.name(),
                                        thresh_type='lower_neq', thresh=0.00)
            
            #tally all the 1s
            wet_cnt = self.rasterlayerstatistics(mask_rlay)['SUM']
            
            #retrieve stats for this iter
            stats_ser = dx.loc[idx[meta_d['resolution'], meta_d['studyArea']], :]
            
            res_d = {'wetArea':wet_cnt * stats_ser['rasterUnitsPerPixelY']*stats_ser['rasterUnitsPerPixelX'],
                    'wetCnt':wet_cnt}
            
                        
            #===================================================================
            # wet volume     
            #===================================================================
            #apply the mask
            rlay_maskd = self.mask_apply(rlay, mask_rlay, logger=log, layname='%s_noGW'%rlay.name())
            
            mask_stats = self.rasterlayerstatistics(rlay_maskd)
            tval = mask_stats['SUM']
            
            res_d.update( {'wetVolume':tval * stats_ser['rasterUnitsPerPixelY']*stats_ser['rasterUnitsPerPixelX'],
                    'tval':tval})
            
            #===================================================================
            # wet mean
            #===================================================================
            assert round(tval/wet_cnt, 3)==round(mask_stats['MEAN'], 3)
            res_d['wetMean'] = mask_stats['MEAN']
            
            
            return res_d
            
 
            
        #=======================================================================
        # execute on stack
        #=======================================================================
        rdx = self.calc_on_layers(func=func, logger=log, dkey=dkey, **kwargs)
        return rdx
        
    def build_gwArea(self,#negative cell count
                    dkey='gwArea',
                    logger=None, **kwargs):
        """TODO: write a test for this"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_gwArea')
        assert dkey=='gwArea'
 
        dx = self.retrieve('rstats')
        #=======================================================================
        # define the function
        #=======================================================================
        def func(rlay, logger=None, meta_d={}):

            if self.rlay_getstats(rlay)['MIN']>=0:
                wet_cnt=0
            else:
            
                #build a mask layer
                mask_rlay = self.mask_build(rlay, logger=logger, layname='%s_neg_mask'%rlay.name(),
                                            thresh_type='upper_neq', thresh=0.0)
                
                #tally all the 1s
                wet_cnt = self.rasterlayerstatistics(mask_rlay)['SUM']
            
            #retrieve stats for this iter
            stats_ser = dx.loc[idx[meta_d['resolution'], meta_d['studyArea']], :]

            return {dkey:wet_cnt * stats_ser['rasterUnitsPerPixelY']*stats_ser['rasterUnitsPerPixelX']}
 
 
        #=======================================================================
        # execute on stack
        #=======================================================================
        return self.calc_on_layers(func=func, logger=log, dkey=dkey, **kwargs)
 
    
    def build_noDataPct(self,
                    dkey='noData_pct',
                    logger=None, **kwargs):
        """TODO: write a test for this (need to run get_drlay)"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_noDataPct')
        assert dkey=='noData_pct'
        
        """need to execute drlays"""
        ndc_dx = self.retrieve('noData_cnt')
        
        dx = self.retrieve('rstats').drop('noData_cnt', axis=1).join(ndc_dx)
 
        serx = dx['noData_cnt2']/dx['cell_cnt']
 
        return serx.rename(dkey).to_frame()
    
    #===========================================================================
    # RASTER DIFFs--------
    #===========================================================================
    def runDiffs(self):#run sequence for difference layer calcs
        
        
        self.retrieve('difrlay_lib')
        
        dx = self.retrieve('rstatsD')
        
        dx = self.retrieve('rmseD')
        
        
 



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
        if out_dir is None: out_dir= self.temp_dir
            
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
            assert isinstance(d0, dict), 'got bad type on reso=%s: %s'%(resolution, type(d0))
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
            self.store_lay_lib(  res_lib, dkey,logger=log)
            
        #=======================================================================
        # wrap
        #=======================================================================
        mstore.removeAllMapLayers()
        assert_lay_lib(res_lib, msg='%s post'%dkey)
        return res_lib
    
    
    def build_rmseD(self,#negative cell count
                    dkey='rmseD',
                    lay_lib=None,
                    logger=None, **kwargs):
 
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_rmseD')
        assert dkey=='rmseD'
 
 
        #=======================================================================
        # retrieve approriate lib
        #=======================================================================
        if lay_lib is None:
            lay_lib = self.retrieve('difrlay_lib')
            
        #=======================================================================
        # calculator
        #=======================================================================
        def func(rlay, logger=None, meta_d={}):
            
            #square the differences
            res_fp = self.rastercalculator(rlay, '{}@1^2'.format(rlay.name()), logger=log)
            
            #get the stats
            sum_sq = self.rasterlayerstatistics(res_fp, logger=log)['SUM']
            
            
            cnt = float(self.rlay_get_cellCnt(res_fp, exclude_nulls=False)) #shouldnt be any nulls
            
 
            return {'rmse':math.sqrt(sum_sq/cnt)}
        
        #=======================================================================
        # execut ethe function on the stack
        #=======================================================================
        serx= self.calc_on_layers(
            func=func, logger=log, dkey=dkey, lay_lib=lay_lib, **kwargs)
        
        return serx
 
    #============================================================================
    # COMBINERS------------
    #============================================================================
    def build_resdx(self, #just combing all the results
        dkey='res_dx',
        
        phase_l=None,
        phase_d = {
            'depth':('rstats', 'wetStats', 'gwArea','noData_cnt', 'noData_pct'),
            'diff':('rstatsD','rmseD'),
            'expo':('rsampStats',)
            
            },

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
        agCn=self.agCn
        if phase_l is None: phase_l=self.phase_l
 
        #clean out
        phase_d = {k:v for k,v in phase_d.items() if k in phase_l}
        
        #=======================================================================
        # reindexing functions
        #=======================================================================
        from agg.hydE.hydE_scripts import cat_reindex as hydE_reindexer
        
        reindex_d = {'expo':lambda x:x.sort_index(sort_remaining=True),
                     'depth':lambda x:x.sort_index(sort_remaining=True),
                     'diff':lambda x:x.sort_index(sort_remaining=True)}
        
        #=======================================================================
        # retrieve and check all
        #=======================================================================
        
        res_d=dict()
        for phase, dki_l in phase_d.items():
            d = dict()
            first = True
            for dki in dki_l:
                raw = self.retrieve(dki)
                
                #use reindexer func for this pahse
                dx = reindex_d[phase](raw) 
                
                #assert np.array_equal(dx.index.names, np.array([resCn, saCn])), dki
                 
                #check consistency within phase
                if first:
                    dx_last = dx.copy()
                    first = False
                else:      
                    assert_index_equal(dx.index, dx_last.index)
                    
    
                d[dki] = dx.sort_index()
            
            res_d[phase] = pd.concat(d, axis=1, names=['dkey', 'stat'])
 
 
        #=======================================================================
        # assemble by type
        #=======================================================================
        first=True
        for phase, dxi in res_d.items():
            #get first
            if first:
                rdx=dxi.copy()
                first=False
                continue
            
            el_d = set_info(dxi.index.names, rdx.index.names)
            #===================================================================
            # simple joins
            #===================================================================
            if np.array_equal(dxi.index, rdx.index):
                rdx = rdx.join(dxi)
                
            #===================================================================
            # expanding join
            #===================================================================
            elif len(el_d['diff_left'])==1:
 
                new_name=list(el_d['diff_left'])[0]
                
                #check the existing indexers match
                skinny_mindex = dxi.index.droplevel(new_name).to_frame().drop_duplicates(
                        ).sort_index().index.reorder_levels(rdx.index.names).sort_values()
                        
                assert_index_equal(skinny_mindex,rdx.index)
                
                #simple join seems to work
                rdx = rdx.join(dxi).sort_index()
                
            elif len(el_d['diff_right'])==1:
                """trying to join a smaller index onto the results which have already been expanded"""
                raise IOError('not implmeented')
            
            else:
                raise IOError(el_d)        
        
 
 
        """
        view(rdx)
        """
        
        rdx = rdx.reorder_levels(self.rcol_l, axis=0).sort_index()
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(rdx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return rdx
    
    def build_layxport(self, #export layers to library
                      dkey='layxport',
                    lib_dir = None, #library directory
                      overwrite=None,
                      compression='med', #keepin this separate from global compression (which applys to all ops)
                      
                      id_params = {}, #additional parameter values to use as indexers in teh library
                      debug_max_len = None,
                      phase_l=None,
                      write=None, logger=None):
        """no cleanup here
        setup for one write per parameterization"""
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey=='layxport'
        if logger is None: logger=self.logger
        log = logger.getChild(dkey)
        if overwrite is None: overwrite=self.overwrite
        if write is None: write=self.write
        if lib_dir is None:
            lib_dir = self.lib_dir
        
        if phase_l is None: phase_l=self.phase_l
            
        assert os.path.exists(lib_dir), lib_dir
        resCn=self.resCn
        saCn=self.saCn
        agCn=self.agCn
        
 
        #=======================================================================
        # setup filepaths4
        #=======================================================================
        
        rlay_dir = os.path.join(lib_dir, 'data', *list(id_params.values()))
 
 
        #=======================================================================
        # re-write raster layers
        #=======================================================================
        """todo: add filesize"""
        ofp_lib = dict()
        cnt0=0
        for phase, (dki, icoln) in  {
            'depth':('drlay_lib', resCn),
             'diff':('difrlay_lib', resCn),
             'expo':('finv_agg_lib',agCn),
            }.items():
            
            #phase selector
            if not phase in phase_l:continue
            
            #===================================================================
            # #add pages
            #===================================================================
            #===================================================================
            # if not icoln in ofp_lib:
            #     ofp_lib[icoln] = dict()
            #===================================================================
            
 
            #===================================================================
            # #retrieve
            #===================================================================
            lay_lib = self.retrieve(dki)
            assert_lay_lib(lay_lib, msg=dki)
            
            #===================================================================
            # #write each layer to file
            #===================================================================
            d=dict()
            cnt=0
            for indx, layer_d in lay_lib.items():
    
                d[indx] = self.store_layer_d(layer_d, dki, logger=log,
                                   write_pick=False, #need to write your own
                                   out_dir = os.path.join(rlay_dir,dki, '%s%04i'%(icoln[0], indx)),
                                   compression=compression, add_subfolders=False,overwrite=overwrite,                               
                                   )
                
                #debug handler
                cnt+=1
                if not debug_max_len is None:
                    if cnt>=debug_max_len:
                        log.warning('cnt>=debug_max_len (%i)... breaking'%debug_max_len)
                        break
            cnt0+=cnt
            #===================================================================
            # compile
            #===================================================================
            #dk_clean = dki.replace('_lib','')
            fp_serx = pd.DataFrame.from_dict(d).stack().swaplevel().rename('fp')
            fp_serx.index.set_names([icoln, saCn], inplace=True)
            
            #===================================================================
            # filesizes
            #===================================================================
            dx = fp_serx.to_frame()
            dx['fp_sizeMB'] = np.nan
            for gkeys, fp in fp_serx.items():
                dx.loc[gkeys, 'fp_sizeMB'] = Path(fp).stat().st_size*1e-6
            
            dx.columns.name='stat'
            assert len(dx)>0
            ofp_lib[dki] = pd.concat({dki:dx}, axis=1, names=['dkey'])
            
        #=======================================================================
        # #concat by indexer
        #=======================================================================
        """because phases have separate indexers but both use this function:
            depths + diffs: indexed by resolution
            expo: indexed by aggLevel
            
        """
        rdx =None
        for dki, dxi in ofp_lib.items():
            if rdx is None: 
                rdx = dxi.copy()
            else:
                rdx = rdx.merge(dxi, how='outer', left_index=True, right_index=True, sort=True)
                
                """
                view(rdx.merge(dxi, how='outer', left_index=True, right_index=True))
                """
 
        rdx = rdx.reorder_levels(self.rcol_l, axis=0).sort_index()
        #=======================================================================
        # write
        #=======================================================================
        log.info('finished writing %i'%cnt0)
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return rdx
        
   

    def write_lib(self,
                  catalog_fp=None,
                  lib_dir = None, #library directory
                  res_dx = None, ldx=None,
                  overwrite=None, logger=None,
                  id_params={},
                  ):
        
        #=======================================================================
        # defautls
        #=======================================================================
        if overwrite is None: overwrite=self.overwrite
        if logger is None: logger=self.logger
        log=logger.getChild('write_lib')
        
        resCn=self.resCn
        saCn=self.saCn
        agCn=self.agCn
        
        
        if lib_dir is None:
            lib_dir = self.lib_dir
            
        if catalog_fp is None: 
            catalog_fp = os.path.join(lib_dir, '%s_run_index.csv'%self.name)
            
        #=======================================================================
        # retrieve
        #=======================================================================
        if res_dx is None:
            res_dx=self.retrieve('res_dx')
           
        if ldx is None: 
            ldx = self.retrieve('layxport', id_params=id_params, lib_dir=lib_dir)
        
        assert_index_equal(res_dx.index, ldx.index)    
        #=======================================================================
        # build catalog--------
        #=======================================================================
        cdx = res_dx.join(ldx)
 
        
        
        #=======================================================================
        # #add additional indexers
        #=======================================================================
        #add singgle value levels from a dictionary
        mdex_df = cdx.index.to_frame().reset_index(drop=True)
        for k,v in id_params.items():
            mdex_df[k] = v
            
            #remove from cols
            if k in cdx.columns.get_level_values(1):
                """TODO: check the values are the same"""
                cdx = cdx.drop(k, level=1, axis=1)
            
        cdx.index = pd.MultiIndex.from_frame(mdex_df)
        
        #=======================================================================
        # write catalog-----
        #=======================================================================
        miss_l = set(cdx.index.names).difference(Catalog.keys)
        assert len(miss_l)==0, 'key mistmatch with catalog worker: %s'%miss_l
        
        with Catalog(catalog_fp=catalog_fp, overwrite=overwrite, logger=log, 
                     index_col=list(range(len(cdx.index.names)))
                                    ) as cat:
            for rkeys, row in cdx.iterrows():
                keys_d = dict(zip(cdx.index.names, rkeys))
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
                       subIndexer='resolution',
                       format='dataFrame',
                       
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
                if not isinstance(rlay, QgsMapLayer):
                    raise Error('got bad type on %s: %s'%(
                        tagi, type(rlay)))
                
                #match the layers crs
                self.qproj.setCrs(rlay.crs())
                
                #execute
                res = func(rlay, logger=log.getChild(tagi),meta_d={'studyArea':studyArea, subIndexer:resolution}, **kwargs)
                
                #post
                
                #assert isinstance(res, dict)                
                d[studyArea]=res
                
            #wrap
            if format=='dataFrame':
                res_d[resolution] = pd.DataFrame(d).T
            else:
                res_d[resolution]=d
            
        #=======================================================================
        # wrap
        #=======================================================================
        if format=='dataFrame':
            rdx = pd.concat(res_d, names=[subIndexer, 'studyArea'])
            
            assert isinstance(rdx, pd.DataFrame)
            
        elif format=='dict':
            rdx = res_d
        else:
            raise IOError(format)
        
        
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
 



 