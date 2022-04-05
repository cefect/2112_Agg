'''
Created on Jan. 18, 2022

@author: cefect



'''
#===============================================================================
# imports--------
#===============================================================================
import os, datetime, math, pickle, copy, random, pprint, shutil
import pandas as pd
import numpy as np

idx = pd.IndexSlice

from hp.basic import set_info

from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo, \
    QgsWkbTypes, QgsRasterLayer, RasterCalc, QgsVectorLayer

import hp.gdal
from hp.exceptions import assert_func


from agg.coms.scripts import Session as agSession
from agg.coms.scripts import Catalog


def serx_smry(serx):
 
    d = dict()
    
    for stat in ['count', 'min', 'mean', 'max', 'sum']:
        f = getattr(serx, stat)
        d['%s_%s'%(serx.name, stat)] = f() 
    return d
 
def get_all_pars(): #generate a matrix of all possible parameter combinations
    pars_lib = copy.deepcopy(Model.pars_lib)
    
    #separate values
    par_vals_d, par_dkey_d = dict(), dict()
    for varnm, d in pars_lib.items():
        par_vals_d[varnm] = d['vals']
        par_dkey_d[varnm] = d['dkey']
        
    
    df = pd.MultiIndex.from_product(par_vals_d.values(), names=par_vals_d.keys()).to_frame().reset_index(drop=True)
    
    lkp_ser = pd.Series({k:par_dkey_d[k] for k in df.columns}, name='dkey')
    lkp_ser.index.name = 'varnm'
    
    """assuming order matches"""
    df.columns = pd.MultiIndex.from_frame(lkp_ser.to_frame().reset_index())
 
    return df.swaplevel(axis=1).sort_index(axis=1)
    """
    view(df)
    """
        
 

class Model(agSession):  # single model run
    """
    
    """
    
    gcn = 'gid'
    scale_cn = 'scale'
    colorMap = 'cool'
    
    #supported parameter values
    pars_lib = {
        'tval_type':{'vals':['uniform', 'rand'],                    'dkey':'tvals'},
        'dscale_meth':{'vals':['centroid', 'none', 'area_split'],   'dkey':'tvals'},
        
        'aggType':{'vals':['none', 'gridded'],                      'dkey':'finv_agg_d'},
        'aggLevel':{'vals':[0, 50, 100, 200],                    'dkey':'finv_agg_d'},
        'sgType':{'vals':['centroids', 'poly'],                         'dkey':'finv_sg_d'},

        
        'severity':{'vals':['hi', 'lo'],                            'dkey':'drlay_d'},
        'resolution':{'vals':[5, 50, 100, 200],                     'dkey':'drlay_d'},
        'resampling':{'vals':['none','Average'],                    'dkey':'drlay_d'},
        'resampStage':{'vals':['none', 'wse', 'depth'],             'dkey':'drlay_d'},
        
        'samp_method':{'vals':['points', 'zonal', 'true_mean'],     'dkey':'rsamps'},
        'zonal_stat':{'vals':['Mean', 'Minimum', 'Maximum', 'none'], 'dkey':'rsamps'},
                
        'vid':{'vals':[49, 798,811, 0],                             'dkey':'vfunc'},


        }
    

    
    def __init__(self,
                 name='hyd',
                 proj_lib={},
                 trim=True,  # whether to apply aois
                 data_retrieve_hndls={}, exit_summary=False,
                 **kwargs):
        
        #===========================================================================
        # HANDLES-----------
        #===========================================================================
        # configure handles
        data_retrieve_hndls = {**data_retrieve_hndls, **{

            # aggregating inventories
            'finv_agg_d':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_layer_d(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_finv_agg(**kwargs),
                },
            
            'finv_agg_mindex':{  # map of aggregated keys to true keys
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_finv_agg_mindex(**kwargs),
                #'build':lambda **kwargs: self.build_finv_agg(**kwargs), 
                    #this function does build.. but calling it with this key makes for ambigious parameterization
                },
            
            'finv_sg_d':{  # sampling geometry
                'compiled':lambda **kwargs:self.load_layer_d(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_sampGeo(**kwargs),
                },
            
            'tvals_raw':{#total asset values (of the raw finv)
                 'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_tvals_raw(**kwargs),
                },
            
            'tvals':{  # total asset values aggregated
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_tvals(**kwargs),
                },
            
            'drlay_d':{ #depth raster layers
                'compiled':lambda **kwargs:self.load_layer_d(**kwargs),
                'build':lambda **kwargs: self.build_drlay(**kwargs),
                },
            
            'rsamps':{  # depth raster samples
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_rsamps(**kwargs),
                },
            
            'rloss':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_rloss(**kwargs),
                },
            'tloss':{  # total losses
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_tloss(**kwargs),
                },
 
            
            }}
        
        super().__init__(exit_summary=exit_summary,
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.proj_lib = proj_lib
        self.trim = trim
 
        
        # checking container
        self.mindex_dtypes = {
                 'studyArea':np.dtype('object'),
                 'id':np.dtype('int64'),
                 self.gcn:np.dtype('int64'),  # both ids are valid
                 'grid_size':np.dtype('int64'),
                 'event':np.dtype('O'),
                 self.scale_cn:np.dtype('int64'),
                         }
        
 

    
    #===========================================================================
    # WRITERS---------
    #===========================================================================
    def write_summary(self, #write a nice model run summary 
                      dkey_l=['tloss', 'rloss', 'rsamps'],
                      out_fp=None,
                      ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('write_summary')
        if out_fp is None: out_fp = os.path.join(self.out_dir, 'summary_%s.xls'%self.longname)
        
        smry_lib = dict()
        log.info('preparing summary on %i dkeys: %s'%(len(dkey_l), dkey_l))
        #=======================================================================
        # retrieve
        #=======================================================================
        
        
        meta_d = self._get_meta()
        
        #=======================================================================
        # summary page
        #=======================================================================
        #clean out
        del meta_d['bk_lib']
        
        for studyArea in self.proj_lib.keys():
            del meta_d[studyArea]
        
        meta_d['dkey_l'] = dkey_l
        
        #force to strings
        d = {k:str(v) for k,v in meta_d.items()}
        
        smry_lib['smry'] = pd.Series(d).rename('').to_frame()
        
        #=======================================================================
        # parameter page
        #=======================================================================
        
        smry_lib['bk_lib'] = pd.DataFrame.from_dict(self.bk_lib).T.stack().to_frame()
        
        #=======================================================================
        # study Areas page
        #=======================================================================
        
        smry_lib['proj_lib'] = pd.DataFrame.from_dict(self.proj_lib).T
        
        #=======================================================================
        # data/results summary
        #=======================================================================
        d = dict()
 
        for dkey in dkey_l:
            
            data = self.retrieve(dkey) #best to call this before finv_agg_mindex
            log.info('collecting stats on \'%s\' w/ %i'%(dkey, len(data)))
            if isinstance(data, pd.DataFrame):
                serx = data.iloc[:, -1] #last one is usually total loss
            else:
                serx=data
                
                
            res_meta_d = dict()
            
            for stat in ['count', 'min', 'mean', 'max', 'sum']:
                f = getattr(serx, stat)
                res_meta_d[stat] = f() 
                
            res_meta_d['shape'] = str(serx.shape)
            
            d[dkey] = res_meta_d
        
 
            
        smry_lib['data_smry'] = pd.DataFrame.from_dict(d).T
        
        #=======================================================================
        # complete datasets
        #=======================================================================
        """only worth writing tloss as this holds all the data"""
        smry_lib = {**smry_lib, **{'tloss':self.retrieve('tloss')}}
        
        #=======================================================================
        # write
        #=======================================================================
        #write a dictionary of dfs
        with pd.ExcelWriter(out_fp) as writer:
            for tabnm, df in smry_lib.items():
                if len(df)>10000:continue
                df.to_excel(writer, sheet_name=tabnm, index=True, header=True)
                
        log.info('wrote %i tabs to %s'%(len(smry_lib), out_fp))
        
        return smry_lib
        
 

        

    
    #===========================================================================
    # DATA CONSTRUCTION-------------
    #===========================================================================
    def run_dataGeneration(self, #convenience for executing data generation calls insequence
                           ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('rDg')
        
        #=======================================================================
        # build the aggregated inventories
        #=======================================================================
        finv_agg_d = self.retrieve('finv_agg_d', logger=log)
        
        #retrieve the index linking back to the raw
        finv_agg_mindex = self.retrieve('finv_agg_mindex', logger=log)
        
        #=======================================================================
        # populate the 'total asset values' on each asset
        #=======================================================================
        #raw values
        finv_true_serx = self.retrieve('tvals_raw', logger=log)
        
        #aggregated values
        finv_agg_serx = self.retrieve('tvals', logger=log)
        
        #=======================================================================
        # build hazard data
        #=======================================================================
        drlay_d = self.retrieve('drlay_d', logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('done')
        
    def build_finv_agg(self,  # build aggregated finvs
                       dkey=None,
                       
                       # control aggregated finv type 
                       aggType=None,
                       aggLevel=None,
                       write=None,
                       **kwargs):
        """
        wrapper for calling more specific aggregated finvs (and their indexer)
            filepaths_dict and indexer are copied
            layer container is not
            
        only the compiled pickles for this top level are required
            (lower levels are still written though)
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_finv_agg')
        if write is None: write=self.write
 
        assert dkey in ['finv_agg_d',
                        #'finv_agg_mindex', #makes specifycing keys tricky... 
                        ], 'bad dkey: \'%s\''%dkey
 
        gcn = self.gcn
        log.info('building \'%s\' ' % (aggType))
 
        aggLevel = int(aggLevel)
        #assert isinstance(aggLevel, int), 'got bad aggLevel type: %s (%s)'%(aggLevel, type(aggLevel))
        #=======================================================================
        # retrive aggregated finvs------
        #=======================================================================
        """these should always be polygons"""
        
        finv_agg_d, finv_gkey_df_d = dict(), dict()
        
        if aggType == 'none':  # see Test_p1_finv_none
            assert aggLevel==0, 'got bad aggLevel: %s'%aggLevel
            res_d = self.sa_get(meth='get_finv_clean', write=False, dkey=dkey, get_lookup=True, **kwargs)
 
        elif aggType == 'gridded':  # see Test_p1_finv_gridded
 
            
            res_d = self.sa_get(meth='get_finv_gridPoly', write=False, dkey=dkey, aggLevel=aggLevel, **kwargs)
 
        else:
            raise Error('not implemented')
        
        # unzip
        for studyArea, d in res_d.items():
            finv_gkey_df_d[studyArea], finv_agg_d[studyArea] = d
            
        #=======================================================================
        # check
        #=======================================================================
        for studyArea, vlay in finv_agg_d.items():
            assert 'Polygon' in QgsWkbTypes().displayString(vlay.wkbType())
        
        #=======================================================================
        # handle layers----
        #=======================================================================
        dkey1 = 'finv_agg_d'
        if write:
            self.store_layer_d(finv_agg_d, dkey1, logger=log)
        
        self.data_d[dkey1] = finv_agg_d
        #=======================================================================
        # handle mindex-------
        #=======================================================================
        """creating a index that maps gridded assets (gid) to their children (id)
        seems nicer to store this as an index
        
        """
        assert len(finv_gkey_df_d) > 0
        
        dkey1 = 'finv_agg_mindex'
        serx = pd.concat(finv_gkey_df_d, verify_integrity=True).iloc[:, 0].sort_index()
 
        serx.index.set_names('studyArea', level=0, inplace=True)
        agg_mindex = serx.to_frame().set_index(gcn, append=True).swaplevel().sort_index().index
 
        self.check_mindex(agg_mindex)

        # save the pickle
        if write:
            self.ofp_d[dkey1] = self.write_pick(agg_mindex,
                           os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey1, self.longname)), logger=log)
        
        # save to data
        self.data_d[dkey1] = copy.deepcopy(agg_mindex)
 
        #=======================================================================
        # return requested data
        #=======================================================================
        """while we build two results here... we need to return the one specified by the user
        the other can still be retrieved from the data_d"""
 
        if dkey == 'finv_agg_d':
            result = finv_agg_d
        elif dkey == 'finv_agg_mindex':
            result = agg_mindex
        
        return result
    
    def build_finv_agg_mindex(self, #special wrapper for finv_agg_mindex requests
                              dkey=None,
                              **kwargs):
        """to force consistent variable handling on these siblings,
        here we just call the other"""
        
        assert dkey=='finv_agg_mindex'
        assert len(kwargs)==0, 'specify kwargs on dkey=finv_agg_d'
        
        #call the sibling
        self.retrieve('finv_agg_d')
        
        return self.data_d[dkey]
        
    def build_tvals_raw(self,  # get the total values on each asset
                    dkey=None,
                    prec=2,
                    tval_type='uniform',  # type for total values
                    normed=True, #normalize per-studArea
                    norm_scale=1e2, #norm scalar
                        #for very big study areas we scale things up to avoid dangerously low values
                    #finv_agg_d=None, #for consistency?
                    mindex=None,
                    write=None,
                    ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_tvals_raw')
        assert dkey == 'tvals_raw'
        if prec is None: prec = self.prec
        if write is None: write=self.write
 
        scale_cn = self.scale_cn
        

        if mindex is None: 
            mindex = self.retrieve('finv_agg_mindex')  # studyArea, id : corresponding gid
 
        #=======================================================================
        # get trues
        #=======================================================================
 
        if tval_type == 'uniform':
            vals = np.full(len(mindex), 1.0)
        elif tval_type == 'rand':
            vals = np.random.random(len(mindex))
 
            
        else:
            raise Error('unrecognized')

        finv_true_serx = pd.Series(vals, index=mindex, name=scale_cn)
 
        self.check_mindex(finv_true_serx.index)
        
        #=======================================================================
        # normalize
        #=======================================================================
        """normalizng to the study area so the total value of all assets per study area always equals 100"""
        if normed:
            log.debug('on %i'%len(finv_true_serx))
            
            l = list()
            for studyArea, gserx in finv_true_serx.groupby(level='studyArea'):
                l.append(gserx.multiply(norm_scale).divide(gserx.sum()))
                
            serx = pd.concat(l)
            
            assert (serx.groupby(level='studyArea').sum().round(self.prec)==norm_scale).all()
                
            finv_true_serx = serx

        

        #=======================================================================
        # wrap
        #=======================================================================
        if write:
            self.ofp_d[dkey] = self.write_pick(finv_true_serx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return finv_true_serx
    
    def build_tvals(self, #downscaling raw asset values onto the aggregated inventory
                    dkey = 'tvals',
                    mindex=None,
                    #finv_agg_d=None,
                    dscale_meth='centroid',
                    tval_type='uniform',
                    write=None,
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_tvals')
        if write is None: write=self.write
        assert dkey == 'tvals'
        if mindex is None: 
            mindex = self.retrieve('finv_agg_mindex')  # studyArea, id : corresponding gid
            
        #generate asset values on the raw
        finv_raw_serx = self.retrieve('tvals_raw', mindex=mindex,tval_type=tval_type, **kwargs)
        
        #=======================================================================
        # check if we are already a true finv
        #=======================================================================
        if np.array_equal(mindex.get_level_values(1).values,mindex.get_level_values(2).values):
            assert dscale_meth=='none', 'finv is already aggregated'
            
        
        #=======================================================================
        # aggregate trues
        #=======================================================================
        if dscale_meth == 'centroid':
            """because the finv_agg_mindex is generated using the centroid intersect
                see StudyArea.get_finv_gridPoly
                se can do a simple groupby to perform this type of downscaling"""
            finv_agg_serx = finv_raw_serx.groupby(level=mindex.names[0:2]).sum()
            
        #no aggregation: base runs
        elif dscale_meth=='none': 
            """this should return the same result as the above groupby"""
            finv_agg_serx = finv_raw_serx.droplevel(2)
 
            
        elif dscale_meth == 'area_split':
            raise Error('dome')
        else:
            raise Error('unrecognized dscale_meth=%s'%dscale_meth)
        
        #=======================================================================
        # wrap
        #=======================================================================
        if write:
            self.ofp_d[dkey] = self.write_pick(finv_agg_serx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return finv_agg_serx
        
    def build_drlay(self, #buidl the depth rasters
                    dkey=None,
 
                   write=None,
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_drlay')
 
 
        if write is None: write=self.write
        assert dkey=='drlay_d'
        
        #=======================================================================
        # retrive rasters per StudyArea------
        #=======================================================================
        """leaving everything on the StudyArea to speed things up"""
        
        #execute
        res_d = self.sa_get(meth='get_drlay', logger=log, dkey=dkey, write=False, **kwargs)
 
        
        #=======================================================================
        # handle layers----
        #=======================================================================
 
        if write:
            self.store_layer_d(res_d, dkey, logger=log)
        
        self.data_d[dkey] = res_d
        
        return res_d
    
    #===========================================================================
    # INTERSECTION----------
    #===========================================================================
    def run_intersection(self, 
                         ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('rInt')
        
        #=======================================================================
        # build sampling geometry
        #=======================================================================
        finv_d = self.retrieve('finv_sg_d', logger=log)
        
        #=======================================================================
        # sample depth raseres
        #=======================================================================
        rsamp_serx = self.retrieve('rsamps', logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('done')
    
    def build_sampGeo(self,  # sampling geometry no each asset
                     dkey='finv_sg_d',
                     sgType='centroids',
                     write=None,
                     finv_agg_d=None,
                     ):
        """
        see test_sampGeo
        """
        #=======================================================================
        # defauts
        #=======================================================================
        assert dkey == 'finv_sg_d'
        log = self.logger.getChild('build_sampGeo')
        if write is None: write=self.write
        
        if finv_agg_d is None: finv_agg_d = self.retrieve('finv_agg_d', write=write)
 
        #=======================================================================
        # loop each polygon layer and build sampling geometry
        #=======================================================================
        log.info('on %i w/ %s' % (len(finv_agg_d), sgType))
        res_d = dict()
        for studyArea, poly_vlay in finv_agg_d.items():
 
            log.info('on %s w/ %i feats' % (studyArea, poly_vlay.dataProvider().featureCount()))
            
            if sgType == 'centroids':
                """works on point layers"""
                sg_vlay = self.centroids(poly_vlay, logger=log)
                
            elif sgType == 'poly':
                assert 'Polygon' in QgsWkbTypes().displayString(poly_vlay.wkbType()), 'bad type on %s' % (studyArea)
                poly_vlay.selectAll()
                sg_vlay = self.saveselectedfeatures(poly_vlay, logger=log)  # just get a copy
                
            else:
                raise Error('not implemented')
            
            #===============================================================
            # wrap
            #===============================================================
            sg_vlay.setName('%s_%s' % (poly_vlay.name(), sgType))
            
            res_d[studyArea] = sg_vlay
        
        #=======================================================================
        # store layers
        #=======================================================================
        if write: ofp_d = self.store_layer_d(res_d, dkey, logger=log)
        
        return res_d
    
    def build_rsamps(self,  # get raster samples for all finvs
                     dkey=None,
                     samp_method='points',  # method for raster sampling
                     finv_sg_d=None, drlay_d=None,
                     write=None,
                     mindex=None, #special catch for test consitency
                     zonal_stat=None, 
                     idfn=None,
                     prec=None,
                     **kwargs):
        """
        keeping these as a dict because each studyArea/event is unique
        """
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey == 'rsamps', dkey
        log = self.logger.getChild('build_rsamps')
 
        if prec is None: prec=self.prec
        if write is None: write=self.write
        
        #get the depth rasters
        if drlay_d is None:drlay_d = self.retrieve('drlay_d')

        #=======================================================================
        # generate depths------
        #=======================================================================
        #=======================================================================
        # simple point-raster sampling
        #=======================================================================
        if samp_method in ['points', 'zonal']:
            #deefaults
            gcn = self.gcn
            if idfn is None: idfn=gcn
            

            
            if finv_sg_d is None: finv_sg_d = self.retrieve('finv_sg_d')
            
            #execute
            res_d = self.sa_get(meth='get_rsamps', logger=log, dkey=dkey, write=False,
                                finv_sg_d=finv_sg_d, idfn=idfn, samp_method=samp_method,
                                zonal_stat=zonal_stat,drlay_d=drlay_d, **kwargs)
            
            #post
            dxind1 = pd.concat(res_d, verify_integrity=True)
 
            res_serx = dxind1.stack(
                dropna=True,  # zero values need to be set per-study area
                ).rename('depth').swaplevel().sort_index(axis=0, level=0, sort_remaining=True) 
                
            res_serx.index.set_names(['studyArea', 'event', gcn], inplace=True)
 
            

        #=======================================================================
        # use mean depths from true assets (for error isolation only)
        #=======================================================================
        elif samp_method == 'true_mean':
            res_serx = self.rsamp_trueMean(dkey,  logger=log, mindex=mindex, drlay_d=drlay_d, **kwargs)
        else:raise Error('bad key')
        
        #=======================================================================
        # post clean
        #=======================================================================
        res_serx = res_serx.round(prec).astype(float)
        #=======================================================================
        # checks
        #=======================================================================
        #type checks
        assert isinstance(res_serx, pd.Series)
        self.check_mindex(res_serx.index)
        
        #value checks
        assert res_serx.notna().all()
        bx = res_serx < 0.0
        if bx.any().any():
            raise Error('got some negative depths')
        
 
        assert res_serx.max()<=100
        
        
 
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.ofp_d[dkey] = self.write_pick(res_serx, os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                            logger=log)
        
        return res_serx
    
    #===========================================================================
    # LOSS CALCS----------
    #===========================================================================
    def run_lossCalcs(self,
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        log=self.logger.getChild('rLoss')
        
        #=======================================================================
        # run depths through vfuncs
        #=======================================================================
        rloss_dx = self.retrieve('rloss', logger=log)
        
        #=======================================================================
        # multiply by total value to get total loss
        #=======================================================================
        tloss_dx = self.retrieve('tloss', logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        
        
    
    def build_rloss(self,  # calculate relative loss from rsamps on each vfunc
                    dkey=None,
                    prec=None,  # precision for RL
                    dxser=None,
                    vid=798,
                    write=None,
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_rloss')
        assert dkey == 'rloss'
        if prec is None: prec = self.prec
        if write is None: write=self.write
        
        
        if dxser is None: dxser = self.retrieve('rsamps')
        log.debug('loaded %i rsamps' % len(dxser))
        #=======================================================================
        # #retrieve child data
        #=======================================================================
 
        # vfuncs
        vfunc = self.build_vfunc(vid=vid, **kwargs)
 
        #=======================================================================
        # loop and calc
        #=======================================================================
        log.info('getting impacts from vfunc %i and %i depths' % (
            vfunc.vid, len(dxser)))
 
 
        ar = vfunc.get_rloss(dxser.values)
        
        assert ar.max() <= 100, '%s returned some RLs above 100' % vfunc.name
 
        
        #=======================================================================
        # combine
        #=======================================================================
        rdxind = dxser.to_frame().join(pd.Series(ar, index=dxser.index, name='rl', dtype=float).round(prec)).astype(float)
        
 
        
        log.info('finished on %s' % str(rdxind.shape))
        
        self.check_mindex(rdxind.index)
 
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.ofp_d[dkey] = self.write_pick(rdxind,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        
        return rdxind
    
    def build_tloss(self,  # get the total loss
                    #data retrieval
                    dkey=None,
                    tv_data = None,
                    rl_dxind = None,
                    
 
                    #control
                    write=None,
                    
                    ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        #scale_cn = self.scale_cn
        log = self.logger.getChild('build_tloss')
        assert dkey == 'tloss'
        if write is None: write=self.write
        #=======================================================================
        # retriever
        #=======================================================================
        
        if tv_data is None: 
            tv_data = self.retrieve('tvals')  # studyArea, id : grid_size : corresponding gid
        
        if rl_dxind is None: 
            #raise Error('the instancing in model_get() is corrupting the data_retrieve_hndls... self is now an empty Model isntance?')
            rl_dxind = self.retrieve('rloss')
        
        #rlnames_d = {lvlName:i for i, lvlName in enumerate(rl_dxind.index.names)}
        
        #assert isinstance(tv_data, pd.Series)
        if isinstance(tv_data, pd.Series):
            """reshaping tval data into multindex for backwards compatability w/ Model runs"""
            tv_data = tv_data.to_frame()
            tv_data = pd.concat({'tvals':tv_data}, axis=1, names=['dkey', 'iter'])
            
        assert np.array_equal(np.array(tv_data.columns.names), np.array(['dkey', 'iter']))
 
        #=======================================================================
        # join tval and rloss
        #=======================================================================
        #promote the rloss into dkeys
        rl_dx = pd.concat(
            {'rloss':rl_dxind.loc[:, ['rl']],
             'rsamps':rl_dxind.loc[:, ['depth']],
             }, axis=1, names = tv_data.columns.names) 
        
        dxind1 = rl_dx.join(tv_data, on=tv_data.index.names)
        
        #assert dxind1[scale_cn].notna().all()
        #=======================================================================
        # calc total loss (loss x scale)
        #=======================================================================
        tl_dxind = dxind1.loc[:, idx['tvals', :]].multiply(dxind1.loc[:, idx['rloss', :]].values, axis=0)
        
        #relable
        tl_dxind.columns = tl_dxind.columns.remove_unused_levels()
        tl_dxind.columns.set_levels([dkey], level=0, inplace=True)
 
        #=======================================================================
        # join back
        #=======================================================================
        #drop the scale to a singiel columns
        #=======================================================================
        # if not scale_cn in dxind1.columns:
        #     dxind1[scale_cn] = dxind1.loc[:, tv_data.columns].mean(axis=1)
        #     dxind1 = dxind1.drop(tv_data.columns, axis=1)
        #=======================================================================
            
        dxind2 = dxind1.join(tl_dxind)
 

        
        #=======================================================================
        # check
        #=======================================================================
        self.check_mindex(dxind2.index)
 
        #=======================================================================
        # wrap
        #=======================================================================
 
        serx1 = dxind2.loc[:, idx['tloss', :]].mean(axis=1).rename('tloss')
            
        mdf = pd.concat({
            'max':serx1.groupby(level='event').max(),
            'count':serx1.groupby(level='event').count(),
            'sum':serx1.groupby(level='event').sum(),
            }, axis=1)
        
        log.info('finished w/ %s and totalLoss: \n%s' % (
            str(dxind2.shape),
            # dxind3.loc[:,tval_colns].sum().astype(np.float32).round(1).to_dict(),
            mdf
            ))

        if write: 
            self.ofp_d[dkey] = self.write_pick(dxind2,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return dxind2
    
  
 
        
    

 
    #===========================================================================
    # HELPERS--------
    #===========================================================================
    
    def vectorize(self, #attach tabular results to vectors
                  dxind,
                  finv_agg_d=None,
                  logger=None):
        """
        creating 1 layer per event
            this means a lot of redundancy on geometry
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('vectorize')
        
        if finv_agg_d is None: finv_agg_d = self.retrieve('finv_agg_d', write=False)
        gcn = self.gcn
        
        log.info('on %i in %i studyAreas'%(len(dxind), len(finv_agg_d)))
        
        assert gcn in dxind.index.names
        #=======================================================================
        # loop and join for each studyArea+event
        #=======================================================================
        cnt = 0
        res_lib = {k:dict() for k in dxind.index.unique('studyArea')}
        
        keyNames = dxind.index.names[0:2]
        for keys, gdf_raw in dxind.groupby(level=keyNames):
            #setup group 
            keys_d = dict(zip(keyNames, keys))
            log.debug(keys_d)
            gdf = gdf_raw.droplevel(keyNames) #drop grouping keys
            assert gdf.index.name==gcn
            
            #get the layer for this
            finv_agg_vlay = finv_agg_d[keys_d['studyArea']]
            df_raw = vlay_get_fdf(finv_agg_vlay).sort_values(gcn).loc[:, [gcn]]
            
            #check key match
            d = set_info(df_raw[gcn], gdf.index)
            assert len(d['diff_left'])==0, 'some results keys not on the layer \n    %s'%d
            
            #join results
            res_df = df_raw.join(gdf, on=gcn).dropna(subset=['tl'], axis=0)
            assert len(res_df)==len(gdf)
            
            #create the layer
            finv_agg_vlay.removeSelection() 
            finv_agg_vlay.selectByIds(res_df.index.tolist()) #select just those with data
            geo_d = vlay_get_geo(finv_agg_vlay, selected=True, logger=log)
            res_vlay= self.vlay_new_df(res_df, geo_d=geo_d, logger=log, 
                                layname='%s_%s_res'%(finv_agg_vlay.name().replace('_', ''), keys_d['event']))
            
            #===================================================================
            # wrap
            #===================================================================
            res_lib[keys[0]][keys[1]]=res_vlay
            cnt+=1
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i layers'%cnt)
        
        return res_lib
            
 
    def load_layer_d(self,  # generic retrival for finv type intermediaries
                  fp=None, dkey=None,
                  **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('load_layer_d.%s' % dkey)
        assert dkey in [  'finv_agg_d', 'finv_sg_d']
        
        vlay_fp_lib = self.load_pick(fp=fp, dkey=dkey)  # {study area: aggLevel : vlay filepath}}
        
        # load layers
        lay_d = dict()
        
        for studyArea, fp in vlay_fp_lib.items():
 
            log.info('loading %s from %s' % (studyArea, fp))
            
            """will throw crs warning"""
            ext = os.path.splitext(os.path.basename(fp))[1]
            #===================================================================
            # vectors
            #===================================================================
            if ext in ['.gpkg', '.geojson']:
            
                lay_d[studyArea] = self.vlay_load(fp, logger=log, 
                                               set_proj_crs=False, #these usually have different crs's
                                                       **kwargs)
            elif ext in ['.tif']:
                lay_d[studyArea] = self.rlay_load(fp, logger=log, 
                                               set_proj_crs=False, #these usually have different crs's
                                                       **kwargs)
            else:
                raise Error('unrecognized filetype: %s'%ext)
        
        return lay_d

    def store_layer_d(self,  # consistent storage of finv containers 
                       layer_d,
                       dkey,
                       out_dir=None,
                       logger=None,
                       write_pick=True,
                       **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('store_layer_d')
        if out_dir is None: out_dir = os.path.join(self.wrk_dir, dkey)
        
        log.info('writing \'%s\' layers to %s' % (dkey, out_dir))
        
        #=======================================================================
        # write layers
        #=======================================================================
        ofp_d = dict()
        cnt = 0
        for studyArea, layer in layer_d.items():
            # setup directory
            od = os.path.join(out_dir, studyArea)
            if not os.path.exists(od):
                os.makedirs(od)
                
            out_fp = os.path.join(od, layer.name())
            #===================================================================
            # write vectors
            #===================================================================
            if isinstance(layer, QgsVectorLayer):
                # write each sstudy area
                ofp_d[studyArea] = self.vlay_write(layer,out_fp,logger=log, **kwargs)
            elif isinstance(layer, QgsRasterLayer):
                ofp_d[studyArea] = self.rlay_write(layer,ofp=out_fp+'.tif',logger=log, **kwargs)
            else:
                raise Error('bad type: %s'%type(layer))
            
            cnt += 1
        
        log.debug('wrote %i layers' % cnt)
        #=======================================================================
        # filepahts container
        #=======================================================================
        
        # save the pickle
        if write_pick:
            """cant pickle vlays... so pickling the filepath"""
            self.ofp_d[dkey] = self.write_pick(ofp_d,
                os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
        # save to data
        self.data_d[dkey] = layer_d
        return ofp_d
    
    def rsamp_trueMean(self,
                       dkey,
 
                       mindex=None,
                       
                       
                       #true controls
                       samp_methodTrue = 'points',
                       finv_sg_true_d=None,
 
                       logger=None,
                       **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('get_rsamp_trueMean')
        
        #=======================================================================
        # retrival
        #=======================================================================
        
 
        if mindex is None: mindex = self.retrieve('finv_agg_mindex')
        #===================================================================
        # get trues-------
        #===================================================================
        """neeed to use all the same functions to derive the trues"""

        
        #=======================================================================
        # sampling geo
        #=======================================================================
        if finv_sg_true_d is None: 
            #load raw inventories
            finv_agg_true_d = self.sa_get(meth='get_finv_clean', write=False, dkey=dkey, get_lookup=False)
            
            #load the sampling geomtry (default is centroids)
            finv_sg_true_d = self.build_sampGeo(finv_agg_d = finv_agg_true_d,write=False)
        

        #=======================================================================
        # sample
        #=======================================================================
 
        true_serx = self.build_rsamps(dkey='rsamps', finv_sg_d = finv_sg_true_d, 
                                      write=False, samp_method=samp_methodTrue, 
                                      idfn='id', **kwargs)        
        
        true_serx.index.set_names('id', level=2, inplace=True)
        #=======================================================================
        # group-----
        #=======================================================================
        #=======================================================================
        # join the gid keys
        #=======================================================================
        jdf = pd.DataFrame(index=mindex).reset_index(drop=False, level=1)
        
        true_serx1 = true_serx.to_frame().join(jdf, how='left').swaplevel(
            ).sort_index().set_index('gid', append=True).swaplevel()
 
 
        #group a series by two levels
        agg_serx = true_serx1.groupby(level=true_serx1.index.names[0:3]).mean().iloc[:,0]
        
        #=======================================================================
        # checks
        #=======================================================================
        #extremes (aggreegated should always be less extreme than raws)
        assert true_serx1.max()[0]>=agg_serx.max()
        assert true_serx1.min()[0]<=agg_serx.min()
        
        #dimensions
        assert len(true_serx1)>=len(agg_serx)
        
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i'%len(agg_serx))
        
 
        return agg_serx

    def sa_get(self,  # spatial tasks on each study area
                       proj_lib=None,
                       meth='get_rsamps',  # method to run
                       dkey=None,
                       write=True,
                       logger=None,
                       **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('run_studyAreas')
        
        if proj_lib is None: proj_lib = self.proj_lib
        
        log.info('on %i \n    %s' % (len(proj_lib), list(proj_lib.keys())))
        
        # assert dkey in ['rsamps', 'finv_agg_d'], 'bad dkey %s'%dkey
        
        #=======================================================================
        # loop and load
        #=======================================================================
        """pass Q handles?
            no.. these are automatically scraped from teh session already"""
        init_kwargs = {k:getattr(self,k) for k in [
            'prec', 'trim', 'out_dir', 'overwrite', 'tag', 'temp_dir',  
            #'logger', automatically scraped from the session
             ]}
        res_d = dict()
        for i, (name, pars_d) in enumerate(proj_lib.items()):
            log.info('%i/%i on %s' % (i + 1, len(proj_lib), name))
            
            with StudyArea(session=self, name=name, 
                           **pars_d, **init_kwargs) as wrkr:
                
                # raw raster samples
                f = getattr(wrkr, meth)
                res_d[name] = f(**kwargs)
                
        #=======================================================================
        # write
        #=======================================================================
        if write:
            """frames between study areas dont have any relation to each other... keeping as dict"""
            raise Error('do we use this?')
            self.ofp_d[dkey] = self.write_pick(res_d, os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
        
        return res_d
    
    def check_mindex(self,  # check names and types
                     mindex,
                     chk_d=None,
                     logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        # if logger is None: logger=self.logger
        # log=logger.getChild('check_mindex')
        if chk_d is None: chk_d = self.mindex_dtypes
        
        #=======================================================================
        # check types and names
        #=======================================================================
        names_d = {lvlName:i for i, lvlName in enumerate(mindex.names)}
        
        assert not None in names_d, 'got unlabled name'
        
        for name, lvl in names_d.items():
 
            assert name in chk_d, 'name \'%s\' not recognized' % name
            assert mindex.get_level_values(lvl).dtype == chk_d[name], \
                'got bad type on \'%s\': %s' % (name, mindex.get_level_values(lvl).dtype.name)
                
        #=======================================================================
        # check index values
        #=======================================================================
        # totality is unique
        bx = mindex.to_frame().duplicated()
        assert not bx.any(), 'got %i/%i dumplicated index entries on %i levels \n    %s' % (
            bx.sum(), len(bx), len(names_d), names_d)
        
        return
    
    def _get_meta(self, #get a dictoinary of metadat for this model
                 ):
        
        d = super()._get_meta()
        
        attns = ['gcn', 'scale_cn']
        
        d = {**d, **{k:getattr(self, k) for k in attns}}
        
        #add project info
        for studyArea, lib in self.proj_lib.items():
            d[studyArea] = lib
            
        #add r un info
        d['date'] = self.start.strftime('%Y-%m-%d %H.%M.%S')
        d['runtime_mins'] = round((datetime.datetime.now() - self.start).total_seconds()/60.0, 3)
 
        
        return d

        
class StudyArea(Model, Qproj):  # spatial work on study areas
    
    finv_fnl = []  # allowable fieldnames for the finv

    def __init__(self,
                 
                 # pars_lib kwargs
                 EPSG=None,
                 finv_fp=None,
 
                 
                 #depth rasters
                 #wd_dir=None,
                 #wd_fp = None,
                 wse_fp_d = None, #{raster tag:fp}
                 dem_fp_d=None, 
                 #drlay_d = None,
                 
                 
                 aoi=None,
                 
                 # control
                 trim=True,
                 idfn='id',  # unique key for assets
                 ** kwargs):
        
        super().__init__(exit_summary=False, **kwargs)
        
        #=======================================================================
        # #set crs
        #=======================================================================
        crs = QgsCoordinateReferenceSystem('EPSG:%i' % EPSG)
        assert crs.isValid()
            
        self.qproj.setCrs(crs)
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.idfn = idfn
        
        self.finv_fnl.append(idfn)
        
        #=======================================================================
        # directories
        #=======================================================================
        self.temp_dir = os.path.join(self.temp_dir, self.name)
        if not os.path.exists(self.temp_dir): os.makedirs(self.temp_dir)
        #=======================================================================
        # load aoi
        #=======================================================================
        if not aoi is None and trim:
            self.load_aoi(aoi)
            
        #=======================================================================
        # load finv
        #=======================================================================
        finv_raw = self.vlay_load(finv_fp, dropZ=True, reproj=True)
        
        # field slice
        fnl = [f.name() for f in finv_raw.fields()]
        drop_l = list(set(fnl).difference(self.finv_fnl))
        if len(drop_l) > 0:
            finv1 = self.deletecolumn(finv_raw, drop_l)
            self.mstore.addMapLayer(finv_raw)
        else:
            finv1 = finv_raw
        
        # spatial slice
        if not self.aoi_vlay is None:
            finv2 = self.slice_aoi(finv1)
            self.mstore.addMapLayer(finv1)
        
        else:
            finv2 = finv1
        
        finv2.setName(finv_raw.name())
            
        # check
        miss_l = set(self.finv_fnl).symmetric_difference([f.name() for f in finv2.fields()])
        assert len(miss_l) == 0, 'unexpected fieldnames on \'%s\' :\n %s' % (miss_l, finv2.name())
  
        #=======================================================================
        # attachments
        #=======================================================================
        self.finv_vlay = finv2
        
        self.wse_fp_d=wse_fp_d
        self.dem_fp_d=dem_fp_d
        #self.drlay_d=drlay_d


        self.logger.info('StudyArea \'%s\' init' % (self.name))
 
    def get_clean_rasterName(self, raster_fn,
                             conv_lib={
                                'LMFRA':{
                                    'AG4_Fr_0050_dep_0116_cmp.tif':'LMFRA_0050yr',
                                    'AG4_Fr_0100_dep_0116_cmp.tif':'LMFRA_0100yr',
                                    'AG4_Fr_0500_dep_0116_cmp.tif':'LMFRA_0500yr',
                                    'AG4_Fr_1000_dep_0116_cmp.tif':'LMFRA_1000yr',
                                    
                                    'AG4_Fr_0050_dep_0116_cmpfnd.tif':'LMFRA_0050yr',
                                    'AG4_Fr_0100_dep_0116_cmpfnd.tif':'LMFRA_0100yr',
                                    'AG4_Fr_0500_dep_0116_cmpfnd.tif':'LMFRA_0500yr',
                                    'AG4_Fr_1000_dep_0116_cmpfnd.tif':'LMFRA_1000yr',
                                    
                                    'AG4_Fr_0500_WL_simu_0415_aoi09_0304.tif':'LM_0500',
                                    'AG4_Fr_0200_WL_simu_0415_aoi09_0304.tif':'LM_0200',
                                    'AG4_Fr_0100_WL_simu_0415_aoi09_0304.tif':'LM_0100'                                    
                                        },
                                
                                'obwb':{
                                    'depth_sB_0500_1218.tif':'obwb_0500yr',
                                    'depth_sB_0100_1218.tif':'obwb_0100yr',
                                    
                                    'depth_sB_0500_1218fnd.tif':'obwb_0500yr',
                                    'depth_sB_0100_1218fnd.tif':'obwb_0100yr',
                                    
                                    'wse_sB_0100_1218_10.tif':'obwb_0100',
                                    'wse_sB_0200_1218_10.tif':'obwb_0200',
                                    'wse_sB_0500_1218_10.tif':'obwb_0500'
                                    
                                    
                                    },
                                'Calgary':{
                                    'IBI_2017CoC_s0_0500_170729_dep_0116.tif':'Calgary_0500yr',
                                    'IBI_2017CoC_s0_0100_170729_dep_0116.tif':'Calgary_0100yr',
                                    
                                    'IBI_2017CoC_s0_0500_170729_dep_0116fnd.tif':'Calgary_0500yr',
                                    'IBI_2017CoC_s0_0100_170729_dep_0116fnd.tif':'Calgary_0100yr',
                                    
                                    'IBI_2017CoC_s0_0500_170729_aoi01_0304.tif':'Calgary_0500',
                                    'IBI_2017CoC_s0_0200_170729_aoi01_0304.tif':'Calgary_0200',
                                    'IBI_2017CoC_s0_0100_170729_aoi01_0304.tif':'Calgary_0100'
                                    },
    
                                        }   
                             ):
        
        rname = raster_fn.replace('.tif', '')
        if self.name in conv_lib:
            if raster_fn in conv_lib[self.name]:
                rname = conv_lib[self.name][raster_fn]
                
        return rname
    
    def get_finv_clean(self,
                       finv_vlay_raw=None,
                       idfn=None,
                       get_lookup=False,
                       gcn=None,
                      ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_finv_clean')
        
        if finv_vlay_raw is None: finv_vlay_raw = self.finv_vlay
        if idfn is None: idfn = self.idfn
 
        #=======================================================================
        # clean finv
        #=======================================================================
        """general pre-cleaning of the finv happens in __init__"""
        
        drop_fnl = set([f.name() for f in finv_vlay_raw.fields()]).difference([idfn])
 
        if len(drop_fnl) > 0:
            finv_vlay = self.deletecolumn(finv_vlay_raw, list(drop_fnl), logger=log)
            self.mstore.addMapLayer(finv_vlay_raw)  
        else:
            finv_vlay = finv_vlay_raw
            
        #=======================================================================
        # wrap
        #=======================================================================
        fnl = [f.name() for f in finv_vlay.fields()]
        assert len(fnl) == 1
        assert idfn in fnl
        
        finv_vlay.setName('%s_clean' % finv_vlay_raw.name())
        log.debug('finished on %s' % finv_vlay.name())
            
        if not get_lookup:
            return finv_vlay
 
        #=======================================================================
        # #build a dummy lookup for consistency w/ get_finv_gridPoly
        #=======================================================================
        if gcn is None: gcn = self.gcn
        
        # rename indexer to match
        finv_vlay1 = self.renameField(finv_vlay, idfn, gcn, logger=log)
        finv_vlay1.setName(finv_vlay.name())
        
        df = vlay_get_fdf(finv_vlay)
        df[gcn] = df[idfn]
        return df.set_index(idfn), finv_vlay1
        
    def get_finv_gridPoly(self,  # get a set of polygon grid finvs (for each grid_size)
                  
                  aggLevel=10,
                  idfn=None,
 
                  overwrite=None,
                  finv_vlay=None,
                  **kwargs):
        """
        
        how do we store an intermitten here?
            study area generates a single layer on local EPSG
            
            Session writes these to a directory

        need a meta table
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_finvs_gridPoly')
        gcn = self.gcn
        if overwrite is None: overwrite = self.overwrite
        
        if idfn is None: idfn = self.idfn
        grid_size = aggLevel
        
        if finv_vlay is None: finv_vlay = self.get_finv_clean(idfn=idfn, **kwargs)
        
        #=======================================================================
        # get points on finv_vlay
        #=======================================================================
        """for clean grid membership.. just using centroids"""
        if not 'Point' in QgsWkbTypes().displayString(finv_vlay.wkbType()):
            finv_pts = self.centroids(finv_vlay, logger=log)
            self.mstore.addMapLayer(finv_pts)
        else:
            finv_pts = finv_vlay
        
        #===================================================================
        # raw grid
        #===================================================================
        log.info('on \'%s\' w/ grid_size=%i' % (finv_vlay.name(), grid_size))
        log.info('    creategrid w/ spacing=%i on  %i' %(grid_size, finv_vlay.dataProvider().featureCount()))
        gvlay1 = self.creategrid(finv_vlay, spacing=grid_size, logger=log)
        self.mstore.addMapLayer(gvlay1)
        
 
        """causing some inconsitent behavior
        #===================================================================
        # active grid cells only
        #===================================================================
 
        #select those w/ some assets
        gvlay1.removeSelection()
        self.createspatialindex(gvlay1)
        log.info('    selecting from grid based on intersect w/ \'%s\''%(finv_vlay.name()))
        self.selectbylocation(gvlay1, finv_vlay, logger=log)
        
        #save these
        gvlay2 = self.saveselectedfeatures(gvlay1, logger=log, output='TEMPORARY_OUTPUT')"""
        
        gvlay2 = gvlay1
        #===================================================================
        # populate/clean fields            
        #===================================================================
        # rename id field
        log.info('   renameField \'id\':\'%s\'' % gcn)
        gvlay3 = self.renameField(gvlay2, 'id', gcn, logger=log)
        self.mstore.addMapLayer(gvlay3)
        
        
        # delete grid dimension fields
        fnl = set([f.name() for f in gvlay3.fields()]).difference([gcn])
        gvlay3b = self.deletecolumn(gvlay3, list(fnl), logger=log)
        # self.mstore.addMapLayer(gvlay3b)
        
        # add the grid_size
        #=======================================================================
        # gvlay4 = self.fieldcalculator(gvlay3b, grid_size, fieldName='grid_size', 
        #                                fieldType='Integer', logger=log)
        #=======================================================================
        gvlay4 = gvlay3b
        
        #===================================================================
        # build refecrence dictionary to true assets
        #===================================================================
        #prep
        self.createspatialindex(finv_pts)
        self.createspatialindex(gvlay4)
        gvlay4.setName('%s_gridded'%finv_vlay.name())
        
        log.info('   joinattributesbylocation \'%s\' (%i) to \'%s\' (%i)'%(
            finv_pts.name(), finv_pts.dataProvider().featureCount(),
            gvlay4.name(), gvlay4.dataProvider().featureCount()))
        
        """very slow for big layers"""
        jd = self.joinattributesbylocation(finv_pts, gvlay4, jvlay_fnl=gcn,
                                           method=1, logger=log,
                                           # predicate='touches',
                 output_nom=os.path.join(self.temp_dir, 'finv_grid_noMatch_%i_%s.gpkg' % (
                                             grid_size, self.longname)))
        
        # check match
        #=======================================================================
        # noMatch_cnt = finv_vlay.dataProvider().featureCount() - jd['JOINED_COUNT']
        # if not noMatch_cnt == 0:
        #     """gid lookup wont work"""
        #     raise Error('for \'%s\' grid_size=%i failed to join  %i/%i assets... wrote non matcherse to \n    %s' % (
        #         self.name, grid_size, noMatch_cnt, finv_vlay.dataProvider().featureCount(), jd['NON_MATCHING']))
        #=======================================================================
                
        jvlay = jd['OUTPUT']
        self.mstore.addMapLayer(jvlay)
        
        df = vlay_get_fdf(jvlay, logger=log).set_index(idfn)
        
        #=======================================================================
        # clear non-matchers
        #=======================================================================
        """3rd time around with this one... I think this approach is cleanest though"""
        log.info('    cleaning \'%s\' w/ %i'%(gvlay4.name(), gvlay4.dataProvider().featureCount()))
        
        grid_df = vlay_get_fdf(gvlay4)
        
        bx = grid_df[gcn].isin(df[gcn].unique())  # just those matching the raws
        
        assert bx.any()
 
        gvlay4.removeSelection()
        gvlay4.selectByIds(grid_df[bx].index.tolist())
        assert gvlay4.selectedFeatureCount() == bx.sum()
        
        gvlay5 = self.saveselectedfeatures(gvlay4, logger=log)
        self.mstore.addMapLayer(gvlay4)
        #===================================================================
        # check against grid points
        #===================================================================
        #=======================================================================
        # """
        # this is an artifact of doing selectbylocation then joinattributesbylocation
        #     sometimes 1 or 2 grid cells are erroneously joined
        #     here we just delete them
        # """
        # gpts_ser = vlay_get_fdf(gvlay4)[gcn]
        #=======================================================================
        
        #=======================================================================
        # set_d = set_info(gpts_ser.values, df[gcn].values)
        # 
        # 
        # if not len(set_d['symmetric_difference'])==0:
        #     del set_d['union']
        #     del set_d['intersection']
        #     log.warning('%s.%i got %i mismatched values... deleteing these grid cells\n   %s'%(
        #         self.name, grid_size, len(set_d['symmetric_difference']), set_d))
        #     
        #     assert len(set_d['diff_right'])==0
        #     
        #     #get matching ids
        #     fid_l = gpts_ser.index[gpts_ser.isin(set_d['diff_left'])].tolist()
        #     gvlay4.removeSelection()
        #     gvlay4.selectByIds(fid_l)
        #     assert gvlay4.selectedFeatureCount()==len(set_d['diff_left'])
        #     
        #     #delete these
        #     gvlay4.invertSelection()
        #     gvlay4 = self.saveselectedfeatures(gvlay4, logger=log)
        #=======================================================================
            
        #===================================================================
        # write
        #===================================================================
        
        gvlay5.setName('finv_gPoly_%i_%s' % (grid_size, self.longname.replace('_', '')))
        """moved onto the session
        if write_grids:
            self.vlay_write(gvlay4, os.path.join(od, gvlay4.name() + '.gpkg'),
                            logger=log)"""
 
        #===================================================================
        # #meta
        #===================================================================
        mcnt_ser = df[gcn].groupby(df[gcn]).count()
        meta_d = {
            'total_cells':gvlay1.dataProvider().featureCount(),
            'active_cells':gvlay5.dataProvider().featureCount(),
            'max_member_cnt':mcnt_ser.max()
            }
 
        #=======================================================================
        # wrap
        #=======================================================================
        
        log.info('finished on %i' % len(df))
 
        return df, gvlay5
    



    def get_drlay(self, #build a depth layer intelligently

                   #raster selection
                   wse_fp_d=None,dem_fp_d=None,
                   severity='hi',  #which wse rastser to select
                   #dem_res=5,
                   
                   #raster downsampling
                   resampStage='none', #which stage of the depth raster calculation to apply the downsampling
                        #none: no resampling happening
                        #wse: resample both rasters before subtraction  
                        #depth: subtract rasters first, then resample the result
                   resampling='none',
                  resolution=5, #0=raw (nicer for variable consistency)
                  base_resolution=5, #resolution of raw data
                   
                   #gen 
                  logger=None, layerName=None,
                  trim=True, #generally just trimming this by default
                   ):
        
        """separate function for 'severity' and 'resolution' (gdalwarp)
        
        todo: add trimming?
        
        zero handling?
        """
           
        #=======================================================================
        # defaults
        #=======================================================================
        if wse_fp_d is None: wse_fp_d = self.wse_fp_d
        if dem_fp_d is None: dem_fp_d=self.dem_fp_d
        if logger is None: logger = self.logger
        if trim is None: trim=self.trim
        temp_dir = os.path.join(self.temp_dir, 'get_drlay')
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        start = datetime.datetime.now()
        log = logger.getChild('get_raster')
        resolution = int(resolution)
        
        """TODO: get this to return somewhere"""
        meta_d = {'resolution':resolution, 'resampStage':resampStage, 'resampling':resampling}
        mstore=QgsMapLayerStore()
        #=======================================================================
        # parameter checks
        #=======================================================================
 
        assert resampStage in ['none', 'depth', 'wse'], resampStage
        #parameter logic
        if resampStage =='none':
            assert resolution==base_resolution, resolution
            assert resampling=='none'
            
        if resampling =='none':
            assert resampStage == 'none'
            
            
        assert resolution>=base_resolution
        assert isinstance(resolution, int)
        
        #=======================================================================
        # #select raster filepaths
        #=======================================================================
        #WSE
        assert severity in wse_fp_d
        wse_raw_fp = wse_fp_d[severity]
        assert os.path.exists(wse_raw_fp)
        
        #DEM
        if resolution in dem_fp_d:
            dem_raw_fp = dem_fp_d[resolution]
        else:
            dem_raw_fp = dem_fp_d[base_resolution] #just take the highest resolution
            
        assert os.path.exists(dem_raw_fp)
        
 
        
        #get names
        baseName = self.get_clean_rasterName(os.path.basename(wse_raw_fp))
        if layerName is None: 
            layerName = baseName + '_%i' % resolution

            
        
        log.info('on %s w/ %s'%(os.path.basename(wse_raw_fp) ,meta_d))
        #=======================================================================
        # trim
        #=======================================================================
        if trim:
 
            """NOTE: this makes the output senstivite to the finv

            e.g., slicing the finv could slightly change the sampling results"""
            vlay = self.finv_vlay
            extents = vlay.extent()
        else:
            wse_raw_rlay = self.get_layer(wse_raw_fp, mstore=mstore)
            extents = wse_raw_rlay.extent()
        
        
        #=======================================================================
        # preCalc 
        #=======================================================================
 
        if resampStage in ['none', 'depth']:
            log.info('warpreproject w/ resolution=%i to %s'%(base_resolution, extents))
            wse_fp = self.warpreproject(wse_raw_fp, compression='none', extents=extents, logger=log,
                                        resolution=base_resolution,
                                        output=os.path.join(temp_dir, 'preCalc_%s'%os.path.basename(wse_raw_fp)))
            
            dem_fp = self.warpreproject(dem_raw_fp, compression='none', extents=extents, logger=log,
                                        resolution=base_resolution,
                                        output=os.path.join(temp_dir, 'preCalc_%s'%os.path.basename(dem_raw_fp)))
 
 
 
        elif resampStage == 'wse':
            log.info('resampling w/ resampStage=%s'%resampStage)
            wse_fp = self.get_resamp(wse_raw_fp, resolution, resampling,  extents=extents, logger=log)
            dem_fp = self.get_resamp(dem_raw_fp, resolution, resampling,  extents=extents, logger=log)
 
        else:
            raise Error('badd resampStage: %s'%resampStage)
        
 
        assert_func(lambda:self.rlay_check_match(wse_fp, dem_fp), msg='dem and wse dont match')
        #=======================================================================
        # subtraction
        #=======================================================================
        log.info('building RasterCalc')
        with RasterCalc(wse_fp, name='dep', session=self, logger=log,out_dir=self.temp_dir,) as wrkr:
            
            wse_rlay = wrkr.ref_lay #loaded during init
            dtm_rlay = wrkr.load(dem_fp)
            
            #===================================================================
            # setup
            #===================================================================
            
            entries_d = {k:wrkr._rCalcEntry(v) for k,v in {'top':wse_rlay, 'bottom':dtm_rlay}.items()}
            formula = '%s - %s'%(entries_d['top'].ref, entries_d['bottom'].ref)
            
            #===================================================================
            # execute subtraction
            #===================================================================
            log.info('executing %s'%formula)
            dep_fp1 = wrkr.rcalc(formula, layname=baseName)
            
            #===================================================================
            # #treat negatives
            #===================================================================
            stats_d = self.rasterlayerstatistics(dep_fp1)
            if stats_d['MIN']<0:
 
                entries_d = {k:wrkr._rCalcEntry(v) for k,v in {'dep':dep_fp1}.items()}
                formula = '{dep} * ({dep} >= 0)'.format(**{k:v.ref for k,v in entries_d.items()})
                log.info('executing for negatives %s'%formula)
                dep_fp1 = wrkr.rcalc(formula, layname=baseName, 
                                 ofp=os.path.join(os.path.dirname(dep_fp1),os.path.basename(dep_fp1).replace('.tif', '_posi.tif'))
                                 )

        
        
        #=======================================================================
        # post-downsample
        #=======================================================================
        if resampStage =='depth':
            log.info('resampling w/ resampStage=%s'%resampStage)
 
            dep_fp2 = self.get_resamp(dep_fp1, resolution, resampling,  extents=extents, logger=log)
            
        else:
            dep_fp2 = dep_fp1
            
 
 
        
            
        #=======================================================================
        # check
        #=======================================================================
        rlay = self.rlay_load(dep_fp2,logger=log)
 
        #stats_d = self.get_rasterstats(rlay)
        #assert stats_d['resolution'] == resolution
        assert rlay.crs()==self.qproj.crs()
        assert int(self.rlay_get_resolution(rlay))==resolution

        
        
        #=======================================================================
        # wrap
        #=======================================================================
        rlay.setName(layerName)
        mstore.removeAllMapLayers()
        tdelta = datetime.datetime.now() - start
        
        log.info('finished in %s on \'%s\' (%i x %i = %i)'%(tdelta,
            rlay.name(), rlay.width(), rlay.height(), rlay.width()*rlay.height()))
        
        
        
        return rlay
    
    def get_resamp(self, fp_raw, resolution, resampling,  
                extents=None,
                logger=None,
                ):
        
    
        
        if resolution == 0:
            assert resampling == 'none', 'got resampling method for resolution=raw'
            wd_fp = fp_raw
        else:
            #===================================================================
            # defaults
            #===================================================================
            if logger is None: logger=self.logger
            log = logger.getChild('get_resamp')
            
            log.info('downsampling \'%s\' w/ resolution=%i and resampling=%s' % (
                os.path.basename(fp_raw).replace('.tif', ''), resolution, resampling))

            #===================================================================
            # execute
            #===================================================================
            ofp = os.path.join(self.temp_dir, os.path.basename(fp_raw).replace('.tif', '') + '_warp%i.tif' % resolution)
            wd_fp = self.warpreproject(
                fp_raw, output=ofp, 
                resolution=resolution, resampling=resampling, 
                compression='none', crsOut=self.qproj.crs(), extents=extents, 
                logger=log)
            
        return wd_fp 
                   

    def get_rsamps(self,  # sample a set of rastsers withon a single finv
                   
                   finv_sg_d=None, drlay_d=None,
                   idfn=None,
                   logger=None,
                   
                   samp_method='points',
                   zonal_stat='Mean',  # stats to use for zonal. 2=mean
                   prec=None,
                   #**kwargs,
                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('get_rsamps')
        
        # if finv_vlay_raw is None: finv_vlay_raw=self.finv_vlay
        if idfn is None: idfn = self.idfn
        if prec is None: prec = self.prec
 
        #=======================================================================
        # precheck finv
        #=======================================================================
        
        finv_vlay_raw = finv_sg_d[self.name]
        
        
        if samp_method == 'points':
            assert 'Point' in QgsWkbTypes().displayString(finv_vlay_raw.wkbType())
        elif samp_method == 'zonal':
            assert 'Polygon' in QgsWkbTypes().displayString(finv_vlay_raw.wkbType())

        else:raise Error('bad key')
            
        assert idfn in [f.name() for f in finv_vlay_raw.fields()], 'missing \'%s\' in %s'%(idfn, finv_vlay_raw.name())
        #=======================================================================
        # clean finv
        #=======================================================================
        """general pre-cleaning of the finv happens in __init__"""
        
        drop_fnl = set([f.name() for f in finv_vlay_raw.fields()]).difference([idfn])
 
        if len(drop_fnl) > 0:
            finv_vlay = self.deletecolumn(finv_vlay_raw, list(drop_fnl), logger=log)
            self.mstore.addMapLayer(finv_vlay)  # keep the raw alive
        else:
            finv_vlay = finv_vlay_raw
            
        assert [f.name() for f in finv_vlay.fields()] == [idfn]
        
        #=======================================================================
        # prechekc depth raster
        #=======================================================================
        rlay = drlay_d[self.name]
        assert isinstance(rlay, QgsRasterLayer)
        
        rname = rlay.name()
        
        #=======================================================================
        # loop and sample--------
        #=======================================================================
 
        #===================================================================
        # sample
        #===================================================================
        if samp_method == 'points':
            vlay_samps = self.rastersampling(finv_vlay, rlay, logger=log, pfx='samp_')
        
        elif samp_method == 'zonal':
            vlay_samps = self.zonalstatistics(finv_vlay, rlay, logger=log, pfx='samp_', stat=zonal_stat)
        else:
            raise Error('not impleented')
        #===================================================================
        # post           
        #===================================================================
        self.mstore.addMapLayer(vlay_samps)
        # change column names
        df = vlay_get_fdf(vlay_samps, logger=log)
        df = df.rename(columns={df.columns[1]:rname})
        
        # force type
        assert idfn in df.columns, 'missing key \'%s\' on %s' % (idfn, finv_vlay.name())
        df.loc[:, idfn] = df[idfn].astype(np.int64)
        
        assert df[idfn].is_unique, finv_vlay.name()
 
        df = df.set_index(idfn).sort_index()
        #=======================================================================
        # fill zeros
        #=======================================================================
        res_df = df.fillna(0.0)
        #=======================================================================
        # wrap
        #=======================================================================

        assert res_df.index.is_unique
        
        log.info('finished on %s and %i rasters w/ %i/%i dry' % (
            finv_vlay.name(), 1, res_df.isna().sum().sum(), res_df.size))
        
        return res_df
    
    #===========================================================================
    # def __exit__(self,  # destructor
    #              * args, **kwargs):
    #     
    #     print('__exit__ on studyArea')
    #     super().__exit__(*args, **kwargs)  # initilzie teh baseclass
    #===========================================================================
    
class ModelStoch(Model):
    def __init__(self,
                 modelID = 0, #unique model ID for the catalog
                 iters=10, #number of stochastic iterations
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.modelID=modelID
        self.iters=iters
        
 
        
    def write_lib(self, #writing pickle w/ metadata
                  lib_dir = None, #library directory
                  mindex = None,
                  overwrite=None,
                  modelID=None,
                  cat_d = {}, #passthrough atts to write in the index
                  dkey='tloss',
                  ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('write_lib')
        if overwrite is None: overwrite=self.overwrite
        if lib_dir is None:
            lib_dir = os.path.join(self.work_dir, 'lib', self.name)
 
        assert os.path.exists(lib_dir), lib_dir
        if modelID is None: modelID=self.modelID
        
        modTag = '%03i_%s'%(modelID, self.longname)
        #=======================================================================
        # setup filepaths4
        #=======================================================================
        catalog_fp = os.path.join(lib_dir, 'model_run_index.csv')
        vlay_dir = os.path.join(lib_dir, 'vlays', modTag)
        
        #clear any existing library methods
        """needs to be before all the writes"""
        if os.path.exists(catalog_fp):
            with Catalog(catalog_fp) as wrkr:
                wrkr.remove_model(modelID)
                
        #=======================================================================
        # retrieve
        #=======================================================================
        tl_dx = self.retrieve(dkey) #best to call this before finv_agg_mindex
        if mindex is None: mindex = self.retrieve('finv_agg_mindex')
        
        
        
        
        """no! this is disaggrigation which is ambigious
        best to levave this for the analysis phase (and just pass the index)
        self.reindex_to_raw(tl_dx)"""
        #=======================================================================
        # write to csv
        #=======================================================================
        """not part of the library
        out_fp = os.path.join(self.out_dir, '%s_tloss.csv'%self.longname)
        tl_dx.to_csv(out_fp)"""

        
        #=======================================================================
        # write vectors
        #=======================================================================
        """here we copy each aggregated vector layer into a special directory in the libary
        these filepaths are then stored in teh model pickle"""
        #setup
        if not os.path.exists(vlay_dir):os.makedirs(vlay_dir)
        
        #retrieve
        finv_agg_d = self.retrieve('finv_agg_d')
        
        #write each layer into the directory
        ofp_d = self.store_layer_d(finv_agg_d, 'finv_agg_d', out_dir=vlay_dir, logger=log, write_pick=False)
        
        #=======================================================================
        # build meta
        #=======================================================================
        meta_d = self._get_meta()
        res_meta_d = serx_smry(tl_dx.loc[:, idx[dkey, :]].mean(axis=1).rename(dkey))
        
        meta_d = {**meta_d, **res_meta_d}
        #=======================================================================
        # add to library
        #=======================================================================
        out_fp = os.path.join(lib_dir, '%s.pickle'%modTag)
        
        meta_d = {**meta_d, **{'pick_fp':out_fp, 'vlay_dir':vlay_dir}}
        
        d = {'name':self.name, 'tag':self.tag,  
             'meta_d':meta_d, 'tloss':tl_dx, 'finv_agg_mindex':mindex, 'finv_agg_d':ofp_d, 'vlay_dir':vlay_dir}
        
        self.write_pick(d, out_fp, overwrite=overwrite, logger=log)
        
        
        
        #=======================================================================
        # update catalog
        #=======================================================================
        #kwargs from meta
        cat_d.update({k:meta_d[k] for k in ['modelID', 'name', 'tag', 'date', 'pick_fp', 'vlay_dir', 'runtime_mins', 'out_dir', 'iters']})
        
 
        cat_d = {**cat_d, **res_meta_d}
        cat_d['pick_keys'] = str(list(d.keys()))
        
        with Catalog(catalog_fp) as wrkr:
            wrkr.add_entry(cat_d)
        
            
        
        #=======================================================================
        # write 
        #=======================================================================
 
        
        log.info('updated catalog %s'%catalog_fp)
        
        return catalog_fp
        
    def build_tvals(self, #stochastic calculation of tvals
                    dkey='tvals',
                    mindex=None, 
                    #finv_agg_d=None,
                    tval_type='rand',  # type for total values
                    iters=None, write=None,
                    **kwargs): 
        
        #=======================================================================
        # defaults
        #=======================================================================
        if write is None: write=self.write
        log=self.logger.getChild('build_tvals')
        if mindex is None: 
            mindex = self.retrieve('finv_agg_mindex')  # studyArea, id : corresponding gid
        
        #if finv_agg_d is None: finv_agg_d = self.retrieve('finv_agg_d')
        if iters is None: iters=self.iters
        
        assert dkey=='tvals'
        #=======================================================================
        # setup pars
        #=======================================================================
        if tval_type == 'uniform':
            iters = 1
 
        #=======================================================================
        # execute children
        #=======================================================================
        res_d = self.model_retrieve(dkey, tval_type=tval_type,
                               mindex=mindex,iters=iters,
                               #finv_agg_d=finv_agg_d, 
                               logger=log, **kwargs)
        
        #=======================================================================
        # assemble and collapse
        #=======================================================================
        dxind = pd.concat(res_d, axis=1)
        
        dx = pd.concat({dkey:dxind},axis=1, names=['dkey', 'iter']) 
        
        #=======================================================================
        # write
        #=======================================================================
        log.info('finished w/ %s'%str(dx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(dx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return dx
    
 
 
        
    
    def model_retrieve(self,
                       dkey='tvals',  # method to run
 
                       iters=None,
                       logger=None,
                       **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('model_get')
        if iters is None: iters=self.iters
 
        
        log.info('calling \'%s\' on %i' %(dkey, iters))
 
        #=======================================================================
        # loop and load
        #=======================================================================

        init_kwargs = {k:getattr(self,k) for k in [
            'name', 'prec', 'trim', 'out_dir', 'overwrite', 'temp_dir', 'bk_lib', 'write']}
        res_d = dict()
        for i in range(iters):
            log.info('%i/%i' % (i + 1, iters))
            
            with Model(session=self, tag='%s_%i'%(self.tag, i),  **init_kwargs) as wrkr:
 
                res_d[i] = wrkr.retrieve(dkey, write=False, **kwargs)
                
 
        
        return res_d
    
    def _get_meta(self, #get a dictoinary of metadat for this model
                 ):
        
        d = super()._get_meta()
 
        
        attns = ['iters', 'modelID']
        
 
        return {**d, **{k:getattr(self, k) for k in attns}}
    
 

