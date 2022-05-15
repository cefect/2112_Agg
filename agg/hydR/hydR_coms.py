'''
Created on May 15, 2022

@author: cefect
'''


#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, sys
import qgis.core
from qgis.core import QgsRasterLayer, QgsMapLayerStore
import pandas as pd
import numpy as np
 

idx = pd.IndexSlice
from hp.exceptions import Error
from hp.pd import get_bx_multiVal, view
 
from hp.Q import assert_rlay_equal, QgsMapLayer
from hp.basic import set_info
from agg.hyd.hscripts import Model

class RRcoms(Model):
    resCn='resolution'
    ridn='rawid'
    agCn='aggLevel'
    saCn='studyArea'
    
    id_params=dict()
    
    def __init__(self,
                  lib_dir=None,
                 **kwargs):
        
        super().__init__(**kwargs)
                
                
        if lib_dir is None:
            lib_dir = os.path.join(self.work_dir, 'lib', self.name)
        #assert os.path.exists(lib_dir), lib_dir
        self.lib_dir=lib_dir
        
    def store_lay_lib(self,  res_lib,dkey,
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
            out_dir=os.path.join(out_dir, 'r%i' % resolution)
            ofp_lib[resolution] = self.store_layer_d(layer_d, dkey, logger=log, 
                write_pick=False, #need to write your own
                out_dir=out_dir)
        
        #=======================================================================
        # #write the pick
        #=======================================================================
        self.ofp_d[dkey] = self.write_pick(ofp_lib, 
            os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)

class Catalog(object): #handling the simulation index and library
    df=None
    keys = ['resolution', 'studyArea', 'downSampling', 'dsampStage', 'severity',
            'aggType', 'samp_method', 'aggLevel']
    cols = ['dkey', 'stat']
 

    
    def __init__(self, 
                 catalog_fp='fp', 
                 logger=None,
                 overwrite=True,
                 index_col=list(range(5)),
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
        """not using this anymore"""
        self.cat_colns = []
        
        #=======================================================================
        # load existing
        #=======================================================================
        if os.path.exists(catalog_fp):
            self.df = pd.read_csv(catalog_fp, 
                                  index_col=index_col,
                                  header = [0,1],
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
        
        
        
        #=======================================================================
        # #check columns
        #=======================================================================
        assert not None in df.columns.names
        miss_l = set(self.cols).symmetric_difference(df.columns.names)
        assert len(miss_l)==0, miss_l
        
        assert isinstance(df.columns, pd.MultiIndex)
        
 
        #=======================================================================
        # #check index
        #=======================================================================
        assert not None in df.index.names
        #=======================================================================
        # assert df[self.idn].is_unique
        # assert 'int' in df[self.idn].dtype.name
        #=======================================================================
        assert isinstance(df.index, pd.MultiIndex)
        
        miss_l = set(df.index.names).difference(self.keys)
        assert len(miss_l)==0, miss_l
 
        
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
        
        """making this more flexible
        keys = self.keys.copy()"""
        keys = list(keys_d.keys())

        log.debug('w/ %i'%len(serx))
        #check mandatory columns are there
        miss_l = set(self.cat_colns).difference(serx.index.get_level_values(1))
        assert len(miss_l)==0, 'got %i missing cols: %s'%(len(miss_l), miss_l)
        
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
        
    def get_dkey_fp(self, #build a pickle by dkey
                   dkey='', 
                   dx_raw=None, #catalog frame
                   
                   #parmaeters
                   pick_indexers=tuple(), #map of how the pickels are indexed
                   #id_params={}, #additional indexers identfying this run
                   #defaults
                   logger=None,
                   **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_pick_%s'%dkey)
        if dx_raw is None: dx_raw=self.get()
        ln1 = pick_indexers[0]
        #=======================================================================
        # precheck
        #=======================================================================
        assert dkey in dx_raw.columns.get_level_values(0), dkey
        
        #=======================================================================
        # #indexers implied by this selection
        # keys_s = set(pick_indexers).union(id_params.keys())
        # 
        # #compoared to indexers found on the catalog
        # miss_l = set(keys_s).difference(dx_raw.index.names)
        # assert len(miss_l)==0
        #=======================================================================
        
        #=======================================================================
        # prep columns
        #=======================================================================
        
        #slice to just this data
        serx1 = dx_raw.loc[:,idx[dkey, 'fp']]
        
 
        #=======================================================================
        # prep index
        #=======================================================================
        
        #=======================================================================
        # bx = get_bx_multiVal(serx0, id_params, matchOn='index', log=log)
        # 
        # serx1 = serx0[bx].droplevel(list(id_params.keys()))
        #=======================================================================
        
        assert serx1.is_unique
        
        assert len(serx1)>0
        
        #=======================================================================
        # collapse to dict
        #=======================================================================
        res_d = dict()
        for studyArea, gserx in serx1.groupby(level=ln1):
            d =  gserx.droplevel(level=ln1).to_dict()
            
            #check these
            for k,fp in d.items():
                assert os.path.exists(fp), 'bad fp on %s.%s: \n    %s'%(studyArea, k, fp)
                
            res_d[studyArea] = d
            
            
        log.info('got %i'%len(serx1))
        
        return res_d
        
 
 
        
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
                if not isinstance(lay, QgsMapLayer):
                    raise AssertionError('bad type on %s.%s: %s\n'%(
                        k0,k1, type(lay))+msg)