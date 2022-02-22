'''
Created on Jan. 18, 2022

@author: cefect



'''
#===============================================================================
# imports--------
#===============================================================================
import os, datetime, math, pickle, copy, random, pprint
import pandas as pd
import numpy as np

idx = pd.IndexSlice

from hp.basic import set_info

from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo, \
    QgsWkbTypes




from agg.coms.scripts import Session as agSession


def serx_smry(serx):
 
    d = dict()
    
    for stat in ['count', 'min', 'mean', 'max', 'sum']:
        f = getattr(serx, stat)
        d['%s_%s'%(serx.name, stat)] = f() 
    return d
 
def get_all_pars(): #generate a matrix of all possible parameter combinations
    pars_lib = Model.pars_lib
    raise Error('create a multindex from multiplication')

    

class Model(agSession):  # single model run
    """
    
    """
    
    gcn = 'gid'
    scale_cn = 'scale'
    colorMap = 'cool'
    
    #supported parameter values
    pars_lib = {
        'tval_type':['uniform', 'rand'],
        'severity':['hi', 'lo'],
        'aggType':['none', 'gridded'],
        'aggLevel':[50, 200],
        
        
        }
    
    def __init__(self,
                 name='hyd',
                 proj_lib={},
                 trim=True,  # whether to apply aois
                 aggType='none',
                 **kwargs):
        
        #===========================================================================
        # HANDLES-----------
        #===========================================================================
        # configure handles
        data_retrieve_hndls = {

            # aggregating inventories
            'finv_agg_d':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_finv_lib(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_finv_agg(**kwargs),
                },
            
            'finv_agg_mindex':{  # map of aggregated keys to true keys
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_finv_agg(**kwargs),
                },
            
            'finv_sg_d':{  # sampling geometry
                'compiled':lambda **kwargs:self.load_finv_lib(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_sampGeo(**kwargs),
                },
            
            'tvals':{  # total asset values (series)
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_tvals(**kwargs),
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
            'errs':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_errs(**kwargs),
                }
            
            }
        
        super().__init__(
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.proj_lib = proj_lib
        self.trim = trim
        self.aggType = aggType
        
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
    def write_lib(self, #writing pickle w/ metadata
                  lib_dir = None, #library directory
                  mindex = None,
                  overwrite=None,
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
        
        catalog_fp = os.path.join(lib_dir, 'catalog.csv')
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
        out_fp = os.path.join(self.out_dir, '%s_tloss.csv'%self.longname)
        tl_dx.to_csv(out_fp)
        #=======================================================================
        # write vectors
        #=======================================================================
        """here we copy each aggregated vector layer into a special directory in the libary
        these filepaths are then stored in teh model pickle"""
        #setup
        dkey = 'finv_agg_d'
        vlay_dir = os.path.join(lib_dir, 'vlays', self.longname)
        if not os.path.exists(vlay_dir):os.makedirs(vlay_dir)
        
        #retrieve
        finv_agg_d = self.retrieve(dkey)
        
        #write each layer into the directory
        ofp_d = self.store_finv_lib(finv_agg_d, dkey, out_dir=vlay_dir, logger=log, write_pick=False)
 
        
        #=======================================================================
        # build meta
        #=======================================================================
        meta_d = self._get_meta()
        res_meta_d = serx_smry(tl_dx['rl'])
        
        meta_d = {**meta_d, **res_meta_d}
        #=======================================================================
        # add to library
        #=======================================================================
        out_fp = os.path.join(lib_dir, '%s.pickle'%self.longname)
        
        meta_d = {**meta_d, **{'pick_fp':out_fp, 'vlay_dir':vlay_dir}}
        
        d = {'name':self.name, 'tag':self.tag,  
             'meta_d':meta_d, 'tloss':tl_dx, 'finv_agg_mindex':mindex, 'finv_agg_d':ofp_d, 'vlay_dir':vlay_dir}
        
        self.write_pick(d, out_fp, overwrite=overwrite, logger=log)
        
        
        
        #=======================================================================
        # update catalog
        #=======================================================================
        cat_d = {k:meta_d[k] for k in ['name', 'tag', 'date', 'pick_fp', 'vlay_dir', 'runtime_mins', 'out_dir']}
        cat_d = {**cat_d, **res_meta_d}
        cat_d['pick_keys'] = str(list(d.keys()))
        
        df = pd.Series(cat_d).to_frame().T
        
        df.to_csv(catalog_fp, mode='a', header = not os.path.exists(catalog_fp), index=False)
        
        #=======================================================================
        # write sumary frame
        #=======================================================================
 
        
        log.info('updated catalog %s'%catalog_fp)
        
        return catalog_fp
    
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
            if isinstance(data, pd.DataFrame):
                serx = data.iloc[:, -1] #last one is usually total loss
            else:
                serx=data
            res_meta_d = dict()
            
            for stat in ['count', 'min', 'mean', 'max', 'sum']:
                f = getattr(serx, stat)
                res_meta_d[stat] = f() 
            
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
                df.to_excel(writer, sheet_name=tabnm, index=True, header=True)
                
        log.info('wrote %i tabs to %s'%(len(smry_lib), out_fp))
        
        return smry_lib
        
 

        

    
    #===========================================================================
    # DATA CONSTRUCTION-------------
    #===========================================================================
    def build_finv_agg(self,  # build aggregated finvs
                       dkey=None,
                       
                       # control aggregated finv type 
                       aggType=None,
                       aggLevel=None,
                       write=True,
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
        assert dkey in ['finv_agg_d',
                        #'finv_agg_mindex', #makes specifycing keys tricky... 
                        ]
        if aggType is None: aggType = self.aggType
        gcn = self.gcn
        log.info('building \'%s\' ' % (aggType))
        
        #=======================================================================
        # retrive aggregated finvs------
        #=======================================================================
        """these should always be polygons"""
        
        finv_agg_d, finv_gkey_df_d = dict(), dict()
        
        if aggType == 'none':  # see Test_p1_finv_none
            assert aggLevel is None
            res_d = self.sa_get(meth='get_finv_clean', write=False, dkey=dkey, get_lookup=True, **kwargs)
 
        elif aggType == 'gridded':  # see Test_p1_finv_gridded
            assert isinstance(aggLevel, int)
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
            self.store_finv_lib(finv_agg_d, dkey1, logger=log)
        
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

    def build_tvals(self,  # get the total values on each asset
                    dkey=None,
                    prec=2,
                    tval_type='uniform',  # type for total values
                    finv_agg_d=None,
                    mindex=None,
                    **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_tvals')
        assert dkey == 'tvals'
        if prec is None: prec = self.prec
 
        scale_cn = self.scale_cn
        
        if finv_agg_d is None: finv_agg_d = self.retrieve('finv_agg_d', **kwargs)
        if mindex is None: mindex = self.retrieve('finv_agg_mindex')  # studyArea, id : corresponding gid
 
        #=======================================================================
        # get trues
        #=======================================================================
 
        if tval_type == 'uniform':
            vals = np.full(len(mindex), 1.0)
        elif tval_type == 'rand':
            vals = np.random.random(len(mindex))
            raise Error('need to implement somekind of simulation here')
            
        else:
            raise Error('unrecognized')

        finv_true_serx = pd.Series(vals, index=mindex, name=scale_cn)
 
        self.check_mindex(finv_true_serx.index)
        
        #=======================================================================
        # aggregate trues
        #=======================================================================
        finv_agg_serx = finv_true_serx.groupby(level=mindex.names[0:2]).sum()
 
        self.ofp_d[dkey] = self.write_pick(finv_agg_serx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return finv_agg_serx
  
    def build_sampGeo(self,  # get raster samples for all finvs
                     dkey='finv_sg_d',
                     sgType='centroids',
                     write=True,
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
        if write: ofp_d = self.store_finv_lib(res_d, dkey, logger=log)
        
        return res_d
    
    def build_rsamps(self,  # get raster samples for all finvs
                     dkey=None,
                     method='points',  # method for raster sampling
                     finv_sg_d=None,
                     write=True,
                     mindex=None, #special catch for test consitency
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
        gcn = self.gcn
        
        if finv_sg_d is None: finv_sg_d = self.retrieve('finv_sg_d')
        #=======================================================================
        # child data
        #=======================================================================

        #=======================================================================
        # generate depths------
        #=======================================================================
        #=======================================================================
        # simple point-raster sampling
        #=======================================================================
        if method in ['points', 'zonal']:
            if idfn is None: idfn=gcn
            res_d = self.sa_get(meth='get_rsamps', logger=log, dkey=dkey, write=False,
                                finv_sg_d=finv_sg_d, idfn=idfn, method=method, **kwargs)
            
            dxind1 = pd.concat(res_d, verify_integrity=True)
 
            res_serx = dxind1.stack(
                dropna=True,  # zero values need to be set per-study area
                ).rename('depth').swaplevel().sort_index(axis=0, level=0, sort_remaining=True) 
                
            res_serx.index.set_names(['studyArea', 'event', gcn], inplace=True)
 
            

        #=======================================================================
        # use mean depths from true assets (for error isolation only)
        #=======================================================================
        elif method == 'true_mean':
            res_serx = self.rsamp_trueMean(dkey,  logger=log, mindex=mindex, **kwargs)
 
        res_serx = res_serx.round(prec).astype(float)
        #=======================================================================
        # checks
        #=======================================================================
        assert isinstance(res_serx, pd.Series)
        bx = res_serx < 0.0
        if bx.any().any():
            raise Error('got some negative depths')
        
        self.check_mindex(res_serx.index)
 
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.ofp_d[dkey] = self.write_pick(res_serx, os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                            logger=log)
        
        return res_serx
    
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
                    tv_serx = None,
                    rl_dxind = None,
                    
                    #control
                    write=None,
                    
                    ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        scale_cn = self.scale_cn
        log = self.logger.getChild('build_tloss')
        assert dkey == 'tloss'
        if write is None: write=self.write
        #=======================================================================
        # retriever
        #=======================================================================
        
        if tv_serx is None: tv_serx = self.retrieve('tvals')  # studyArea, id : grid_size : corresponding gid
        
        if rl_dxind is None: rl_dxind = self.retrieve('rloss')
        
        #rlnames_d = {lvlName:i for i, lvlName in enumerate(rl_dxind.index.names)}
        
        #=======================================================================
        # join tval and rloss
        #=======================================================================
        dxind1 = rl_dxind.join(tv_serx, on=tv_serx.index.names)
        
        assert dxind1[scale_cn].notna().all()
        #=======================================================================
        # calc total loss (loss x scale)
        #=======================================================================
        dxind1['tl'] = dxind1['rl'].multiply(dxind1[scale_cn])
        
        #=======================================================================
        # check
        #=======================================================================
        self.check_mindex(dxind1.index)
 
        #=======================================================================
        # wrap
        #=======================================================================
        # reporting
        serx1 = dxind1['tl']
        mdf = pd.concat({
            'max':serx1.groupby(level='event').max(),
            'count':serx1.groupby(level='event').count(),
            'sum':serx1.groupby(level='event').sum(),
            }, axis=1)
        
        log.info('finished w/ %s and totalLoss: \n%s' % (
            str(dxind1.shape),
            # dxind3.loc[:,tval_colns].sum().astype(np.float32).round(1).to_dict(),
            mdf
            ))

        if write: 
            self.ofp_d[dkey] = self.write_pick(dxind1,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return dxind1
    
  
 
        
    

 
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
            
        
            
 
        
        


    def load_finv_lib(self,  # generic retrival for finv type intermediaries
                  fp=None, dkey=None,
                  **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('load_finv_lib.%s' % dkey)
        assert dkey in ['finv_agg_d', 'finv_agg_d', 'finv_sg_d']
        
        vlay_fp_lib = self.load_pick(fp=fp)  # {study area: aggLevel : vlay filepath}}
        
        # load layers
        finv_agg_d = dict()
        
        for studyArea, fp in vlay_fp_lib.items():
 
            log.info('loading %s from %s' % (studyArea, fp))
            
            """will throw crs warning"""
            finv_agg_d[studyArea] = self.vlay_load(fp, logger=log, **kwargs)
        
        return finv_agg_d

    def store_finv_lib(self,  # consistent storage of finv containers 
                       finv_grid_lib,
                       dkey,
                       out_dir=None,
                       logger=None,
                       write_pick=True,
                       ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('store_finv')
        if out_dir is None: out_dir = os.path.join(self.wrk_dir, dkey)
        
        log.info('writing \'%s\' layers to %s' % (dkey, out_dir))
        
        #=======================================================================
        # write layers
        #=======================================================================
        ofp_d = dict()
        cnt = 0
        for studyArea, poly_vlay in finv_grid_lib.items():
            # setup directory
            od = os.path.join(out_dir, studyArea)
            if not os.path.exists(od):
                os.makedirs(od)
                
            # write each sstudy area
            ofp_d[studyArea] = self.vlay_write(poly_vlay,
                    os.path.join(od, poly_vlay.name()),
                    logger=log)
            cnt += 1
        
        log.debug('wrote %i layers' % cnt)
        #=======================================================================
        # filepahts container
        #=======================================================================
        
        # save the pickle
        if write_pick:
            """cant pickle vlays... so pickling the filepath"""
            self.ofp_d[dkey] = self.write_pick(ofp_d,
                os.path.join(out_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
        # save to data
        self.data_d[dkey] = finv_grid_lib
        return ofp_d
    
    def rsamp_trueMean(self,
                       dkey,
 
                       mindex=None,
                       
                       #true controls
                       methodTrue = 'points',
                       finv_sg_true_d=None,
                       sampGeoTrueKwargs = {},
                           logger=None,
                           ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('get_rsamp_trueMean')
 
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
            finv_sg_true_d = self.build_sampGeo(finv_agg_d = finv_agg_true_d,write=False, **sampGeoTrueKwargs)
        

        #=======================================================================
        # sample
        #=======================================================================
 
        true_serx = self.build_rsamps(dkey='rsamps', finv_sg_d = finv_sg_true_d, write=False, method=methodTrue, 
                                      idfn='id')        
        
        true_serx.index.set_names('id', level=2, inplace=True)
        #=======================================================================
        # group-----
        #=======================================================================
        #=======================================================================
        # join the gid keys
        #=======================================================================
        jdf = pd.DataFrame(index=mindex).reset_index(drop=False, level=1)
        true_serx1 = true_serx.to_frame().join(jdf, how='left').swaplevel().sort_index().set_index('gid', append=True).swaplevel()
 
        #=======================================================================
        # group
        #=======================================================================
        #group a series by two levels
        agg_serx = true_serx1.groupby(level=true_serx1.index.names[0:3]).mean().iloc[:,0]
        
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
        res_d = dict()
        for i, (name, pars_d) in enumerate(proj_lib.items()):
            log.info('%i/%i on %s' % (i + 1, len(proj_lib), name))
            
            with StudyArea(session=self, name=name, tag=self.tag, prec=self.prec,
                           trim=self.trim, out_dir=self.out_dir, overwrite=self.overwrite,
                           **pars_d) as wrkr:
                
                # raw raster samples
                f = getattr(wrkr, meth)
                res_d[name] = f(**kwargs)
                
        #=======================================================================
        # write
        #=======================================================================
        if write:
            """frames between study areas dont have any relation to each other... keeping as dict"""

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
                 dem=None,
                 
                 #depth rasters
                 #wd_dir=None,
                 #wd_fp = None,
                 wd_fp_d = None, #{raster tag:fp}
                 
                 
                 aoi=None,
                 
                 # control
                 trim=True,
                 idfn='id',  # unique key for assets
                 ** kwargs):
        
        super().__init__(**kwargs)
        
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
        #self.wd_dir = wd_dir
        #self.wd_fp=wd_fp
        self.wd_fp_d = copy.deepcopy(wd_fp_d)
        self.logger.info('StudyArea \'%s\' init' % (self.name))
 
    def get_clean_rasterName(self, raster_fn,
                             conv_lib={
                                'LMFRA':{
                                    'AG4_Fr_0050_dep_0116_cmp.tif':'0050yr',
                                    'AG4_Fr_0100_dep_0116_cmp.tif':'0100yr',
                                    'AG4_Fr_0500_dep_0116_cmp.tif':'0500yr',
                                    'AG4_Fr_1000_dep_0116_cmp.tif':'1000yr',
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
        log = self.logger.getChild('get_finvs_gridPoly')
        
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
        
        gvlay1 = self.creategrid(finv_vlay, spacing=grid_size, logger=log)
        self.mstore.addMapLayer(gvlay1)
        log.info('    built grid w/ %i' % gvlay1.dataProvider().featureCount())
 
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
        gvlay3 = self.renameField(gvlay2, 'id', gcn, logger=log)
        self.mstore.addMapLayer(gvlay3)
        log.info('    renamed field \'id\':\'%s\'' % gcn)
        
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
        jd = self.joinattributesbylocation(finv_pts, gvlay4, jvlay_fnl=gcn,
                                           method=1, logger=log,
                                           # predicate='touches',
                 output_nom=os.path.join(self.temp_dir, 'finv_grid_noMatch_%i_%s.gpkg' % (
                                             grid_size, self.longname)))
        
        # check match
        noMatch_cnt = finv_vlay.dataProvider().featureCount() - jd['JOINED_COUNT']
        if not noMatch_cnt == 0:
            """gid lookup wont work"""
            raise Error('for \'%s\' grid_size=%i failed to join  %i/%i assets... wrote non matcherse to \n    %s' % (
                self.name, grid_size, noMatch_cnt, finv_vlay.dataProvider().featureCount(), jd['NON_MATCHING']))
                
        jvlay = jd['OUTPUT']
        self.mstore.addMapLayer(jvlay)
        
        df = vlay_get_fdf(jvlay, logger=log).set_index(idfn)
        
        #=======================================================================
        # clear non-matchers
        #=======================================================================
        """3rd time around with this one... I think this approach is cleanest though"""
        
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
 
 

    def get_rsamps(self,  # sample a set of rastsers withon a single finv
                   wd_fp_d=None,
                   finv_sg_d=None,
                   idfn=None,
                   logger=None,
                   severity='hi',
                   method='points',
                   zonal_stats=[2],  # stats to use for zonal. 2=mean
                   prec=None,
                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('get_rsamps')
        if wd_fp_d is None: wd_fp_d = self.wd_fp_d
        # if finv_vlay_raw is None: finv_vlay_raw=self.finv_vlay
        if idfn is None: idfn = self.idfn
        if prec is None: prec = self.prec
        
        #select raster filepath
        assert severity in wd_fp_d
        wd_fp = wd_fp_d[severity]
        
        #=======================================================================
        # precheck
        #=======================================================================
        
        finv_vlay_raw = finv_sg_d[self.name]
        
        assert os.path.exists(wd_fp)
        if method == 'points':
            assert 'Point' in QgsWkbTypes().displayString(finv_vlay_raw.wkbType())
        elif method == 'zonal':
            assert 'Polygon' in QgsWkbTypes().displayString(finv_vlay_raw.wkbType())
            assert isinstance(zonal_stats , list)
            assert len(zonal_stats) == 1
            
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
        # loop and sample
        #=======================================================================
 
        rname = self.get_clean_rasterName(os.path.basename(wd_fp))
 
        
        #===================================================================
        # sample
        #===================================================================
        if method == 'points':
            vlay_samps = self.rastersampling(finv_vlay, wd_fp, logger=log, pfx='samp_')
        
        elif method == 'zonal':
            vlay_samps = self.zonalstatistics(finv_vlay, wd_fp, logger=log, pfx='samp_', stats=zonal_stats)
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
