'''
Created on Feb. 21, 2022

@author: cefect
'''
#===============================================================================
# imports------
#===============================================================================
import os, datetime, math, pickle, copy, random, pprint, gc, math
import matplotlib
import scipy.stats

import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal

idx = pd.IndexSlice




import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#===============================================================================
# custom imports
#===============================================================================
from hp.basic import set_info, get_dict_str
from hp.exceptions import Error, assert_func

from hp.plot import Plotr
from hp.oop import Session
from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo, \
    QgsWkbTypes
    
    
from agg.coms.scripts import Catalog, ErrorCalcs
from agg.hyd.hscripts import HydSession



def get_ax(
        figNumber=0,
        figsize=(4,4),
        tight_layout=False,
        constrained_layout=True,
        ):
    
    if figNumber in plt.get_fignums():
        plt.close()
    
    fig = plt.figure(figNumber,figsize=figsize,
                tight_layout=tight_layout,constrained_layout=constrained_layout,
                )
            
    return fig.add_subplot(111)
 
class ModelAnalysis(HydSession, Qproj, Plotr): #analysis of model results
    
    
    
    #colormap per data type
    colorMap_d = {
        'aggLevel':'cool',
        'dkey_range':'winter',
        'studyArea':'Dark2',
        'modelID':'Pastel1',
        'dsampStage':'Set1',
        'downSampling':'Set2',
        'aggType':'Pastel2',
        'tval_type':'Set1'
        }
    
    def __init__(self,
                 catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hyd2\model_run_index.csv',
                 plt=None,
                 name='analy',
                 modelID_l=None, #optional list for specifying a subset of the model runs
                 #baseID=0, #mase model ID
                 exit_summary=False,
                 **kwargs):
        
        data_retrieve_hndls = {
            'catalog':{
                #probably best not to have a compliled version of this
                'build':lambda **kwargs:self.build_catalog(**kwargs), #
                },
            'outs':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_outs(**kwargs),
                },
            'agg_mindex':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_agg_mindex(**kwargs),
                },
            'trues':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs), #consider checking the baseID
                'build':lambda **kwargs: self.build_trues(**kwargs),
                },
            'deltas':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs), #consider checking the baseID
                'build':lambda **kwargs: self.build_deltas(**kwargs),
                },
            'finv_agg_fps':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs), #only filepaths?
                'build':lambda **kwargs: self.build_finv_agg(**kwargs),
                },
            
            }
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls,name=name,init_plt_d=None,
                         work_dir = r'C:\LS\10_OUT\2112_Agg', exit_summary=exit_summary,
                         **kwargs)
        self.plt=plt
        self.catalog_fp=catalog_fp
        #self.baseID=baseID
        self.modelID_l=modelID_l
    
    #===========================================================================
    # RESULTS ASSEMBLY---------
    #===========================================================================
    def runCompileSuite(self, #conveneince for compilling all the results in order
                        ):
 
        
        cat_df = self.retrieve('catalog')
               
        dx_raw = self.retrieve('outs')
        
        agg_mindex = self.retrieve('agg_mindex')
        
        
        
        true_d = self.retrieve('trues')
        
        
        
    
    def build_catalog(self,
                      dkey='catalog',
                      catalog_fp=None,
                      logger=None,
                      **kwargs):
        if logger is None: logger=self.logger
        assert dkey=='catalog'
        if catalog_fp is None: catalog_fp=self.catalog_fp
        
        return Catalog(catalog_fp=catalog_fp, logger=logger, overwrite=False, **kwargs).get()
    
    def build_agg_mindex(self,
                         dkey='agg_mindex',
    
                        write=None,
                        idn=None, logger=None,
                     **kwargs):
        """
        todo: check against loaded outs?
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_agg_mindex')
        assert dkey=='agg_mindex'
        if write is None: write=self.write
        if idn is None: idn=self.idn
        #=======================================================================
        # pull from piciles
        #=======================================================================
        #get each finv_agg_mindex from the run pickels
        data_d= self.assemble_model_data(dkey='finv_agg_mindex', 
                                         logger=log, write=write, idn=idn,
                                         **kwargs)
        
        #=======================================================================
        # combine
        #=======================================================================
        log.debug('on %s'%data_d.keys())
        
        d = {k: mdex.to_frame().reset_index(drop=True) for k, mdex in data_d.items()}
        
        dx1 = pd.concat(d, names=[idn, 'index'])
        
        mdex = dx1.set_index(['studyArea', 'gid', 'id'], append=True).droplevel('index').index
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(mdex.to_frame().shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(mdex,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return mdex
    
    def build_outs(self, #collecting outputs from multiple model runs
                    dkey='outs',
                         write=None,
                         idn=None,
                         cat_df=None,
                         logger=None,
                         **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_outs')
        assert dkey=='outs'
        if cat_df is None: 
            cat_df = self.retrieve('catalog')
        if write is None: write=self.write
        if idn is None: idn=self.idn
        #=======================================================================
        # pull from piciles
        #=======================================================================
        
        data_d= self.assemble_model_data(dkey='tloss', 
                                         logger=log, write=write, cat_df=cat_df, idn=idn,
                                         **kwargs)
    
    
        #=======================================================================
        # combine
        #=======================================================================
        dx = pd.concat(data_d).sort_index(level=0)
 
            
        dx.index.set_names(idn, level=0, inplace=True)
        
        #join tags
        dx.index = dx.index.to_frame().join(cat_df['tag']).set_index('tag', append=True
                          ).reorder_levels(['modelID', 'tag', 'studyArea', 'event', 'gid'], axis=0).index
        
        """
        view(dx)
        """
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(dx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(dx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return dx
    

    def build_finv_agg(self,
                         dkey='finv_agg_fps',
    
                        write=None,
                        idn=None,
                        logger=None,
                     **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_finv_agg')
        assert dkey=='finv_agg_fps'
        if write is None: write=self.write
        if idn is None: idn=self.idn
        #=======================================================================
        # pull from piciles
        #=======================================================================
        
        fp_lib= self.assemble_model_data(dkey='finv_agg_d', 
                                         logger=log, write=write, idn=idn,
                                         **kwargs)
        
        #=======================================================================
        # check
        #=======================================================================
        cnt = 0
        for mid, d in fp_lib.items():
            for studyArea, fp in d.items():
                assert os.path.exists(fp)
                cnt+=1
                
        
        
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('retrieved %i finv_agg fps'%cnt)
        if write:
            self.ofp_d[dkey] = self.write_pick(fp_lib,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return fp_lib
        
        
    
    def assemble_model_data(self, #collecting outputs from multiple model runs
                   modelID_l=None, #set of modelID's to include (None=load all in the catalog)
                   dkey='outs',
                    
                    cat_df=None,
                     idn=None,
                     write=None,
                     logger=None,
                     ):
        """
        loop t hrough each pickle, open, then retrive the requested dkey
        just collecting into a dx for now... now meta
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('build_%s'%dkey)
        
        if write is None: write=self.write
        if idn is None: idn=self.idn
        if cat_df is None: cat_df = self.retrieve('catalog')
        if modelID_l is None: modelID_l=self.modelID_l
        
        assert idn==Catalog.idn
        #=======================================================================
        # retrieve catalog
        #=======================================================================
        
        
        if modelID_l is None:
            modelID_l= cat_df.index.tolist()
        
        assert len(modelID_l)>0
        #check
        miss_l = set(modelID_l).difference(cat_df.index)
        assert len(miss_l)==0, '%i/%i requested %s not found in catalog:\n    %s'%(
            len(miss_l), len(modelID_l), idn, miss_l)
        
        log.info('on %i'%len(modelID_l))
        #=======================================================================
        # load data from modle results
        #=======================================================================
        data_d = dict()
        for modelID, row in cat_df.loc[cat_df.index.isin(modelID_l),:].iterrows():
            log.info('    on %s.%s'%(modelID, row['tag']))
            
            #check
            pick_keys = eval(row['pick_keys'])
            assert dkey in pick_keys, 'requested dkey not stored in the pickle: \'%s\''%dkey
            
            #load pickel            
            with open(row['pick_fp'], 'rb') as f:
                data = pickle.load(f) 
                data_d[modelID] = data[dkey].copy()
                
                del data
                
        #=======================================================================
        # wrap
        #=======================================================================
        gc.collect()
        return data_d
                

    
    def build_trues(self, #map 'base' model data onto the index of all the  models 
                         
                     baseID_l=[0], #modelID to consider 'true'
                     #modelID_l=None, #optional slicing to specific models
                     dkey='trues',
                     dx_raw=None, agg_mindex=None,
                     
                     idn=None, write=None, logger=None,
                     ):
        """
        
        for direct comparison w/ aggregated model results,
            NOTE the index is expanded to preserve the trues
            these trues will need to be aggregated (groupby) again (method depends on what youre doing)
            
        TODO: check the specified baseID run had an aggLevel=0
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('trues')
        if idn is None: idn=self.idn
        if write is None: write=self.write
        assert dkey == 'trues'
        
        #combined model outputs (on aggregated indexes): {modelID, tag, studyArea, event, gid}
        if dx_raw is None: 
            dx_raw = self.retrieve('outs')
            
        #index relating aggregated ids to raw ids {modelID, studyArea, gid, id}
        if agg_mindex is None:
            agg_mindex = self.retrieve('agg_mindex')
        
        
        
        log.info('on %s'%str(dx_raw.shape))
        
        #=======================================================================
        # check
        #=======================================================================
        #clean out extra index levels for checking
        l = set(dx_raw.index.names).difference(agg_mindex.names)
        chk_mindex = dx_raw.index.droplevel(list(l))
        
        
        assert_func(lambda: self.check_mindex_match_cats(agg_mindex,chk_mindex, glvls = [self.idn, 'studyArea']), 'agg_mindex')
        
        #check the base indicies are there
        miss_l = set(baseID_l).difference(dx_raw.index.unique(idn))
        assert len(miss_l)==0, 'requested baseIDs not loaded: %s'%miss_l
 
        
        miss_l = set(dx_raw.index.unique(idn)).symmetric_difference(agg_mindex.unique(idn))
        assert len(miss_l)==0, '%s mismatch between outs and agg_mindex... recompile?'%idn
        
        #=======================================================================
        # build for each base
        #=======================================================================
        log.info('building %i true sets'%len(baseID_l))
        res_d = dict()
        for i, baseID in enumerate(baseID_l):
            log.info('%i/%i on baseID=%i'%(i+1, len(baseID_l), baseID))
            #=======================================================================
            # get base
            #=======================================================================
            #just the results for the base model
            base_dx = dx_raw.loc[idx[baseID, :, :, :], :].droplevel([idn, 'tag', 'event']).dropna(how='all', axis=1)
            base_dx.index = base_dx.index.remove_unused_levels().set_names('id', level=1)
     
            #=======================================================================
            # expand to all results
            #=======================================================================
            #add the events and tags
            amindex1 = agg_mindex.join(dx_raw.index).reorder_levels(dx_raw.index.names + ['id'])
            
     
            #create a dummy joiner frame
            """need a 2d column index for the column dimensions to be preserved during the join"""
            jdx = pd.DataFrame(index=amindex1, columns=pd.MultiIndex.from_tuples([('a',1)], names=base_dx.columns.names))
            
            #=======================================================================
            # #loop and add base values for each Model
            #=======================================================================
            d = dict()
            err_d = dict()
            for modelID, gdx0 in jdx.groupby(level=idn):
                """
                view(gdx0.sort_index(level=['id']))
                """
                log.debug(modelID)
                #check indicides
                try:
                    assert_index_equal(
                        base_dx.index,
                        gdx0.index.droplevel(['modelID', 'tag', 'gid', 'event']).sortlevel()[0],
                        #check_order=False, #not in 1.1.3 yet 
                        #obj=modelID #not doing anything
                        )
                    
                    #join base data onto this models indicides
                    gdx1 =  gdx0.join(base_dx, on=base_dx.index.names).drop('a', axis=1, level=0)
                    
                    #check
                    assert gdx1.notna().all().all(), 'got %i/%i nulls'%(gdx1.isna().sum().sum(), gdx1.size)
                    d[modelID] = gdx1.copy()
                except Exception as e:
                    err_d[modelID] = e
            
            #report errors
            if len(err_d)>0:
                for mid, msg in err_d.items():
                    log.error('%i: %s'%(mid, msg))
                raise Error('failed join on %i/%i \n    %s'%(
                    len(err_d), len(jdx.index.unique(idn)), list(err_d.keys())))
                    
     
                
            #combine
            dx1 = pd.concat(d.values())
                
     
            #=======================================================================
            # check
            #=======================================================================
            assert dx1.notna().all().all()
            #check columns match
            assert np.array_equal(dx1.columns, base_dx.columns)
            
            #check we still match the aggregated index mapper
            assert np.array_equal(
                dx1.index.sortlevel()[0].to_frame().reset_index(drop=True).drop(['tag','event'], axis=1).drop_duplicates(),
                agg_mindex.sortlevel()[0].to_frame().reset_index(drop=True)
                ), 'result failed to match original mapper'
            
            assert_series_equal(dx1.max(axis=0), base_dx.max(axis=0))
     
            assert_func(lambda: self.check_mindex_match(dx1.index, dx_raw.index), msg='raw vs trues')
            
            res_d[baseID] = dx1.copy()
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on  %i'%len(res_d))
 
 
        if write:
            self.ofp_d[dkey] = self.write_pick(res_d,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return res_d
    
    
    def build_deltas(self,
                     
                     #input data
                     dx_raw=None,
                     true_dx=None, #base values mapped onto all the other models
                     
                     
                     dkey=None,write=None, logger=None,
                     ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        raise Error('not implemented')
        if logger is None: logger=self.logger
        assert dkey=='deltas'
        log = self.logger.getChild(dkey)
        idn=self.idn
        if write is None: write=self.write
 
        
        #=======================================================================
        # retrieve
        #=======================================================================
        if dx_raw is None:  
            dx_raw = self.retrieve('outs')
            
        if true_dx_raw is None:
            true_d = self.retrieve('trues')
            true_dx_raw = true_d[baseID]
 
            
        log.info('on raw: %s and true: %s'%(str(dx_raw.shape), str(true_dx.shape)))
 
         
 
                
 
        
        
 
        
 
    def xxxbuild_errs(self,  # get the errors (gridded - true)
                    dkey=None,
                     prec=None,
                     group_keys=['grid_size', 'studyArea', 'event'],
                     write_meta=True,
 
                    ):
        """
        delta: grid - true
        errRel: delta/true
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_errs')
        assert dkey == 'errs'
        if prec is None: prec = self.prec
        gcn = self.gcn
        scale_cn = self.scale_cn
 
        #=======================================================================
        # retriever
        #=======================================================================
        tl_dx = self.retrieve('tloss')
        
        tlnames_d = {lvlName:i for i, lvlName in enumerate(tl_dx.index.names)}
 
        fgdir_dxind = self.retrieve('finv_agg_mindex')
        
        fgdir_dxind[0] = fgdir_dxind.index.get_level_values('id')  # set for consistency
 
        #=======================================================================
        # group on index
        #=======================================================================
        log.info('on %s' % str(tl_dx.shape))
        res_dx = None
        for ikeys, gdx0 in tl_dx.groupby(level=group_keys, axis=0):
            ikeys_d = dict(zip(group_keys, ikeys))
            res_lib = {k:dict() for k in tl_dx.columns.unique(0)}
            #===================================================================
            # group on columns
            #===================================================================
            for ckeys, gdx1 in gdx0.groupby(level=gdx0.columns.names, axis=1):
                ckeys_d = dict(zip(gdx0.columns.names, ckeys))
                
                log.debug('on %s and %s' % (ikeys_d, ckeys_d))
 
                #===================================================================
                # get trues--------
                #===================================================================
                
                true_dx0 = tl_dx.loc[idx[0, ikeys_d['studyArea'], ikeys_d['event'],:], gdx1.columns]
                
                #===============================================================
                # join true keys to gid
                #===============================================================
                # relabel to true ids
                true_dx0.index.set_names('id', level=tlnames_d['gid'], inplace=True)
                
                # get true ids (many ids to 1 gid))
                id_gid_df = fgdir_dxind.loc[idx[ikeys_d['studyArea'],:], ikeys_d['grid_size']].rename(gcn).to_frame()
                id_gid_dx = pd.concat([id_gid_df], keys=['expo', 'gid'], axis=1)
                
                if not id_gid_dx.index.is_unique:
                    # id_gid_ser.to_frame().loc[id_gid_ser.index.duplicated(keep=False), :]
                    raise Error('bad index on %s' % ikeys_d)
                
                # join gids
                true_dxind1 = true_dx0.join(id_gid_dx, on=['studyArea', 'id']).sort_index().droplevel(0, axis=1)

                #===============================================================
                # summarize by type
                #===============================================================
                # get totals per gid
                gb = true_dxind1.groupby(gcn)
                if ckeys_d['lossType'] == 'tl':  # true values are sum of each child
                    true_df0 = gb.sum()
                elif ckeys_d['lossType'] == 'rl':  # true values are the average of family
                    true_df0 = gb.mean()
                elif ckeys_d['vid'] == 'depth':
                    true_df0 = gb.mean()
                elif ckeys_d['vid'] == scale_cn: 
                    true_df0 = gb.sum()
                else:
                    raise Error('bad lossType')

                assert true_df0.index.is_unique
                
                # expand index
                true_dx = pd.concat([true_df0], keys=['true', true_df0.columns[0]], axis=1) 
 
                # join back to gridded
                gdx2 = gdx1.join(true_dx, on=gcn, how='outer')
                
                #===========================================================
                # get gridded-------
                #===========================================================
                
                # check index
                miss_l = set(gdx2.index.get_level_values(gcn)).difference(true_dx.index.get_level_values(gcn))
                assert len(miss_l) == 0, 'failed to join back some trues'

                """from here... we're only dealing w/ 2 columns... building a simpler calc frame"""
                
                gdf = gdx2.droplevel(1, axis=1).droplevel(group_keys, axis=0)
                gdf0 = gdf.rename(columns={gdf.columns[0]:'grid'})
                
                #===================================================================
                # error calcs
                #===================================================================
                # delta (grid - true)
                gdf1 = gdf0.join(gdf0['grid'].subtract(gdf0['true']).rename('delta'))
                
                # relative (grid-true / true)
                gdf2 = gdf1.join(gdf1['delta'].divide(gdf1['true']).fillna(0).rename('errRel'))
 
                #===============================================================
                # clean
                #===============================================================
                # join back index
                rdxind1 = gdx1.droplevel(0, axis=1).join(gdf2, on=gcn).drop(gdx1.columns.get_level_values(1)[0], axis=1)
                
                # check
                assert rdxind1.notna().all().all()
                
                assert not ckeys[1] in res_lib[ckeys[0]]
                
                # promote
                res_lib[ckeys[0]][ckeys[1]] = rdxind1
  
            #===================================================================
            # wrap index loop-----
            #===================================================================
            
            #===================================================================
            # assemble column loops
            #===================================================================
            d2 = dict()
            names = [tl_dx.columns.names[1], 'metric']
            for k0, d in res_lib.items():
                d2[k0] = pd.concat(d, axis=1, names=names)
                
            rdx = pd.concat(d2, axis=1, names=[tl_dx.columns.names[0]] + names)
            
            #===================================================================
            # append
            #===================================================================
            if res_dx is None:
                res_dx = rdx
            else:
                res_dx = res_dx.append(rdx)
 
        #=======================================================================
        # wrap------
        #=======================================================================
        #=======================================================================
        # promote vid to index
        #=======================================================================
        """
        treating meta columns (id_cnt, depth) as 'vid'
        makes for more flexible data maniuplation
            although.. .we are duplicating lots of values now
        """
        
        res_dx1 = res_dx.drop('expo', level=0, axis=1)
        
        # promote column values to index
        res_dx2 = res_dx1.stack(level=1).swaplevel().sort_index()
        
        # pull out and expand the exposure
        exp_dx1 = res_dx.loc[:, idx['expo',:,:]].droplevel(0, axis=1)
        
        # exp_dx2 = pd.concat([exp_dx1, exp_dx1], keys = res_dx1.columns.unique(0), axis=1)
        
        # join back
        res_dx3 = res_dx2.join(exp_dx1, on=res_dx1.index.names).sort_index(axis=0)
        
        """
        view(res_dx3.droplevel('gid', axis=0).index.to_frame().drop_duplicates())
        view(res_dx3)
        """

        #===================================================================
        # meta
        #===================================================================
        gb = res_dx.groupby(level=group_keys)
         
        mdx = pd.concat({'max':gb.max(), 'count':gb.count(), 'sum':gb.sum()}, axis=1)
        
        if write_meta:
            ofp = os.path.join(self.out_dir, 'build_errs_smry_%s.csv' % self.longname)
            if os.path.exists(ofp):assert self.overwrite
            mdx.to_csv(ofp)
            log.info('wrote %s to %s' % (str(mdx.shape), ofp))

        log.info('finished w/ %s and totalErrors: \n%s' % (
            str(res_dx.shape), mdx))
 
        #=======================================================================
        # write
        #=======================================================================
        
        self.ofp_d[dkey] = self.write_pick(res_dx3,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return res_dx3
 

 
                  

    
    
    #===========================================================================
    # ANALYSIS WRITERS---------
    #===========================================================================

 
        
 
    def xxxwrite_loss_smry(self,  # write statistcs on total loss grouped by grid_size, studyArea, and event
                    
                   # data control   
                    dkey='tloss',
                    # lossType='tl', #loss types to generate layers for
                    gkeys=[ 'studyArea', 'event', 'grid_size'],
                    
                    # output config
                    write=True,
                    out_dir=None,
                    ):
 
        """not an intermediate result.. jsut some summary stats
        any additional analysis should be done on the raw data
        """
        #=======================================================================
        # defaults
        #=======================================================================
        scale_cn = self.scale_cn
        log = self.logger.getChild('write_loss_smry')
        assert dkey == 'tloss'
        if out_dir is None:
            out_dir = os.path.join(self.out_dir, 'errs')
 
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        # errors
        dx_raw = self.retrieve(dkey)
        
        """
        view(self.retrieve('errs'))
        view(dx_raw)
        """
 
        log.info('on %i for \'%s\'' % (len(dx_raw), gkeys))
        #=======================================================================
        # calc group stats-----
        #=======================================================================
        rlib = dict()
        #=======================================================================
        # loss values
        #=======================================================================
        
        for lossType in dx_raw.columns.unique('lossType'):
            if lossType == 'expo':continue
            dxind1 = dx_raw.loc[:, idx[lossType,:]].droplevel(0, axis=1)
            # mdex = dxind1.index
            
            gbo = dxind1.groupby(level=gkeys)
            
            # loop and get each from the grouper
            d = dict()
            for statName in ['sum', 'mean', 'min', 'max']:
                d[statName] = getattr(gbo, statName)()
                
            # collect
            
            #=======================================================================
            # errors
            #=======================================================================
            """could also do this with the 'errs' data set... but simpler to just re-calc the totals here"""
            err_df = None
            for keys, gdf in gbo:
                keys_d = dict(zip(gkeys, keys))
                
                if keys_d['grid_size'] == 0: continue
                
                # get trues
                """a bit awkward as our key order has changed"""
                true_gdf = dxind1.loc[idx[0, keys_d['studyArea'], keys_d['event']],:]
     
                # calc delta (gridded - true)
                eser1 = gdf.sum() - true_gdf.sum()
     
                # handle results
                """couldnt figure out a nice way to handle this... just collecting in frame"""
                ival_ser = gdf.index.droplevel('gid').to_frame().reset_index(drop=True).iloc[0,:]
                
                eser2 = pd.concat([eser1, ival_ser])
                
                if err_df is None:
                    err_df = eser2.to_frame().T
                    
                else:
                    err_df = err_df.append(eser2, ignore_index=True)
            
            # collect
            d['delta'] = pd.DataFrame(err_df.loc[:, gdf.columns].values,
                index=pd.MultiIndex.from_frame(err_df.loc[:, gkeys]),
                columns=gdf.columns)
            
            rlib[lossType] = pd.concat(d, axis=1).swaplevel(axis=1).sort_index(axis=1)
        
        #=======================================================================
        # meta stats 
        #=======================================================================
        meta_d = dict()
        d = dict()
        dindex2 = dx_raw.loc[:, idx['expo',:]].droplevel(0, axis=1)
        
        d['count'] = dindex2['depth'].groupby(level=gkeys).count()
        
        #=======================================================================
        # depth stats
        #=======================================================================
        gbo = dindex2['depth'].groupby(level=gkeys)
        
        d['dry_cnt'] = gbo.agg(lambda x: x.eq(0).sum())
        
        d['wet_cnt'] = gbo.agg(lambda x: x.ne(0).sum())
 
        # loop and get each from the grouper
        for statName in ['mean', 'min', 'max', 'var']:
            d[statName] = getattr(gbo, statName)()
            
        meta_d['depth'] = pd.concat(d, axis=1)
        #=======================================================================
        # asset count stats
        #=======================================================================
        gbo = dindex2[scale_cn].groupby(level=gkeys)
        
        d = dict()
        
        d['mode'] = gbo.agg(lambda x:x.value_counts().index[0])
        for statName in ['mean', 'min', 'max', 'sum']:
            d[statName] = getattr(gbo, statName)()
 
        meta_d['assets'] = pd.concat(d, axis=1)
        
        #=======================================================================
        # collect all
        #=======================================================================
        rlib['meta'] = pd.concat(meta_d, axis=1)
        
        rdx = pd.concat(rlib, axis=1, names=['cat', 'var', 'stat'])
        
        #=======================================================================
        # write
        #=======================================================================
        log.info('finished w/ %s' % str(rdx.shape))
        if write:
            ofp = os.path.join(self.out_dir, 'lossSmry_%i_%s.csv' % (
                  len(dx_raw), self.longname))
            
            if os.path.exists(ofp): assert self.overwrite
            
            rdx.to_csv(ofp)
            
            log.info('wrote %s to %s' % (str(rdx.shape), ofp))
        
        return rdx
            
        """
        view(rdx)
        mindex.names
        view(dx_raw)
        """
    
    def write_suite_smry(self, #write a summary of the model suite
                         
                         #data
                         dx_raw=None,
                         true_dx_raw=None,
                         modelID_l=None, #models to include
                         baseID=0,
                         
                         #control
                         agg_d = {'rloss':'mean', 'rsamps': 'mean', 'tvals':'sum', 'tloss':'sum'}, #aggregation methods
                         ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('suite_smry')
        
        
        #=======================================================================
        # retrieve
        #=======================================================================
        idn = self.idn
        #if baseID is None: baseID=self.baseID
 
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
            
        if true_dx_raw is None:
            true_d = self.retrieve('trues')
            true_dx_raw = true_d[baseID]
            
        if modelID_l is None:
            modelID_l = dx_raw.index.unique(0).tolist()
            
        #assert baseID == modelID_l[0]
        
        
        
        #=======================================================================
        # prep data
        #=======================================================================
        dx = dx_raw.loc[idx[modelID_l, :, :, :, :], :]
        true_dx = true_dx_raw.loc[idx[modelID_l, :,:, :, :, :], :]
        
        #=======================================================================
        # loop and calc summary
        #=======================================================================
        log.info('on %i models w/ %s'%(len(modelID_l), str(dx_raw.shape)))
        
        res_lib = dict()
        true_gb = true_dx.groupby(level=0, axis=0)
        for gkey, gdx0 in dx.groupby(level=0, axis=0): #loop by axis data
            #===================================================================
            # setup
            #===================================================================            
            keys_d = dict(zip([dx.index.names[0]], [gkey]))
            tgdx0 = true_gb.get_group(gkey)
            
            
 
            #===================================================================
            # by column
            #===================================================================
            res_d = dict()
            for coln, aggMethod in agg_d.items():
                keys_d['coln'] = coln
                log.debug('on %s w/ AggMethod: %s'%(keys_d, aggMethod))
                
                #trim the data to this column
                """taking the mean of all iterations for now"""
                gdx1 = gdx0.loc[:, idx[coln, :]].mean(axis=1)
                tgdx1 = tgdx0.loc[:, idx[coln, :]].mean(axis=1)
                
                """todo: refactor and incorpoarte into plot_compare_mat"""
                #===============================================================
                # totals----
                #===============================================================
                
                #===============================================================
                # rser = pd.Series({
                #     'pred':getattr(gdx1, aggMethod)(), 
                #     'true':getattr(tgdx1, aggMethod)()
                #     })
                #===============================================================
                rser = pd.Series()
                #===================================================================
                # bias
                #===================================================================
                
                """works for multi and single iterations"""

                
                #rser['bias'] = rser['pred']/rser['true']
                
                
                
                #===============================================================
                # per-asset-----
                #===============================================================
                #aggregate trues
                tgdx2 = getattr(tgdx1.groupby(level=[gdx1.index.names]), aggMethod)()
                tgdx2 = tgdx2.reorder_levels(gdx1.index.names).sort_index()
                
                assert_index_equal(gdx1.index, tgdx2.index)
                
                #===============================================================
                # mean errors
                #===============================================================
                ecWrkr = ErrorCalcs(pred_ser=gdx1, true_ser=tgdx2, logger=log)
                

                
                rser['bias'] = ecWrkr.get_bias() 
                
                if not rser['bias']==1.0:
                    print(rser['bias'])
                                    
                rser['meanError'] = ecWrkr.get_meanError()
                #(gdx1 - tgdx2).sum()/len(gdx1)
                
                raise Error('stopped here')
                rser['meanErrorAbs'] = (gdx1 - tgdx2).abs().sum()/len(gdx1)
                rser['RMSE'] = math.sqrt(((gdx1 - tgdx2)**2).sum()/len(gdx1))
                
                #===============================================================
                # confusion
                #===============================================================
                if coln == 'rsamps':
                    df = pd.DataFrame({'true':tgdx2.values, 'pred':gdx1.values})
                    cm_df, cm_dx = self.get_confusion(df, logger=log)
                    
                    rser = rser.append(cm_dx.droplevel(['pred', 'true']).iloc[:,0])
 
                    
 
                #===============================================================
                # wrap
                #===============================================================
                res_d[coln] = rser.astype(float).round(3)
                
            #===================================================================
            # combine
            #===================================================================
            res_lib[gkey] = pd.concat(res_d, names=['var', 'metric'])
            
        #=======================================================================
        # wrap
        #=======================================================================
        rdx = pd.concat(res_lib, axis=1, names=[dx.index.names[0]]).T
        
        #=======================================================================
        # write
        #=======================================================================
        ofp = os.path.join(self.out_dir, 'suite_%s.csv'%self.longname)
        
        rdx.to_csv(ofp)
        
        log.info('wrote %s to \n    %s'%(str(rdx.shape), ofp))
        
        return ofp
                
 
        
        
    
    def write_resvlay(self,  #attach some data to the gridded finv
                  # data control   
                  modelID_l=None,
                  dx_raw=None,finv_agg_fps=None,
                  dkey='tloss',
                  stats = ['mean', 'min', 'max', 'var'], #stats for compressing iter data
                    
                    # output config
 
                    out_dir=None,
                    ):
        """
        folder: tloss_spatial
            sub-folder: studyArea
                sub-sub-folder: event
                    file: one per grid size 
        """
            
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('write_resvlay')
 
        if out_dir is None:out_dir = self.out_dir
        idn=self.idn
        gcn = 'gid'
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        # errors
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
 
        
        # geometry
        if finv_agg_fps is None:
            finv_agg_fps = self.retrieve('finv_agg_fps')
            
        #=======================================================================
        # slice
        #=======================================================================
        if modelID_l is None:
            modelID_l = dx_raw.index.unique(idn).tolist()
            
 
        
        
        dxind = dx_raw.loc[idx[modelID_l, :, :, :, :], idx[dkey, :]].droplevel(0, axis=1)
        
 
        
        #=======================================================================
        # loop and write--------
        #=======================================================================
        log.info('on \'%s\' w/ %i' % (dkey, len(dx_raw)))
        #=======================================================================
        # loop on study area
        #=======================================================================
        ofp_l = list()
        gnames = [idn, 'studyArea', 'event']
        for keys, gdx0 in dxind.groupby(level=gnames):
 
            keys_d = dict(zip(gnames, keys))
            mstore = QgsMapLayerStore()
            log.info(keys_d)
            
            
            #===================================================================
            # prep
            #===================================================================
            assert len(gdx0.index.unique('tag'))==1
            tag = gdx0.index.unique('tag')[0]
            keys_d['tag'] = tag
            
            #drop extra indicies
            gdf1 = gdx0.droplevel([0,1,2,3])
                
            #===================================================================
            # retrieve spatial data
            #===================================================================
            # get vlay
            finv_fp = finv_agg_fps[keys_d[idn]][keys_d['studyArea']]
            finv_vlay = self.vlay_load(finv_fp, logger=log, set_proj_crs=True)
            mstore.addMapLayer(finv_vlay)
            
            #get geo
            geo_d = vlay_get_geo(finv_vlay)
            
            #===================================================================
            # re-key
            #===================================================================
            fid_gid_d = vlay_get_fdata(finv_vlay, fieldn=gcn)
            
            #chekc keys
            miss_l = set(gdf1.index).difference(fid_gid_d.values())
            if not len(miss_l)==0:
                raise Error('missing %i/%i keys on %s'%(len(miss_l), len(gdf1), keys_d))
            
            #rekey data
            fid_ser = pd.Series({v:k for k,v in fid_gid_d.items()}, name='fid')
            gdf2 = gdf1.join(fid_ser).set_index('fid')
            
            """
            gdf2.columns
            """
            
            #===================================================================
            # compute stats
            #===================================================================
            d = {'mean':gdf2.mean(axis=1)}
            
            if len(gdf2.columns)>0:
                d = {k:getattr(gdf2, k)(axis=1) for k in stats}
                
            gdf3 = pd.concat(d.values(), keys=d.keys(), axis=1).sort_index()
            """
            view(gdx1)
            """
 
            #===================================================================
            # build layer
            #===================================================================
            layname = 'm' + '_'.join([str(e).replace('_', '') for e in keys_d.values()])
            vlay = self.vlay_new_df(gdf3, geo_d=geo_d, layname=layname, logger=log,
                                    crs=finv_vlay.crs(),  # session crs does not match studyAreas
                                    )
            mstore.addMapLayer(vlay)
            #===================================================================
            # write layer
            #===================================================================
            # get output directory
            od = os.path.join(out_dir, tag)
 
            if not os.path.exists(od):
                os.makedirs(od)
                
            s = '_'.join([str(e) for e in keys_d.values()])
            ofp = os.path.join(od, self.longname + '_%s_'%dkey + s + '.gpkg') 
 
            ofp_l.append(self.vlay_write(vlay, ofp, logger=log))
            
            #===================================================================
            # wrap
            #===================================================================
            mstore.removeAllMapLayers()
            
        #=======================================================================
        # write meta
        #=======================================================================
        log.info('finished on %i' % len(ofp_l))
        
 
        return ofp_l
            

    
    #===========================================================================
    # PLOTTERS-------------
    #===========================================================================
    def plot_model_smry(self, #plot a summary of data on one model
                        modelID,
                        dx_raw=None,
                        
                        #plot config
                        plot_rown='dkey',
                        plot_coln='event',
                        #plot_colr = 'event',
                        xlims_d = {'rsamps':(0,5)}, #specyfing limits per row
                        
                        #errorbars
                        #qhi=0.99, qlo=0.01,
                        
                         #plot style
                         drop_zeros=True,
                         colorMap=None,
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_model_smry_%i'%modelID)
        if dx_raw is None: dx_raw = self.retrieve('outs')
        if colorMap is None: colorMap=self.colorMap_d['dkey_range']
        
        #=======================================================================
        # data prep
        #=======================================================================
        dx = dx_raw.loc[idx[modelID, :, :, :, :], :].droplevel([0,1])
        mdex = dx.index
        log.info('on %s'%str(dx.shape))
        
        tag = dx_raw.loc[idx[modelID, :, :, :, :], :].index.remove_unused_levels().unique('tag')[0]
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = dx.columns.unique(plot_rown).tolist() 
 
        fig, ax_d = self.get_matrix_fig(row_keys,col_keys, 
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='none', sharex='row',  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        fig.suptitle('Model Summary for \'%s\''%(tag))
        
        # get colors
        #cvals = dx_raw.index.unique(plot_colr)
        cvals = ['min', 'mean', 'max']
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        #===================================================================
        # loop and plot
        #===================================================================
        for col_key, gdx1 in dx.groupby(level=[plot_coln]):
            keys_d = {plot_coln:col_key}
            
            for row_key, gdx2 in gdx1.groupby(level=[plot_rown], axis=1):
                keys_d[plot_rown] = row_key
                ax = ax_d[row_key][col_key]
                
                #===============================================================
                # prep data
                #===============================================================
                gb = gdx2.groupby('dkey', axis=1)
                
                d = {k:getattr(gb, k)() for k in cvals}
                err_df = pd.concat(d, axis=1).droplevel(axis=1, level=1)
                
                bx = err_df==0
                if drop_zeros:                    
                    err_df = err_df.loc[~bx.all(axis=1), :]
                    
                if keys_d['dkey'] in xlims_d:
                    xlims = xlims_d[keys_d['dkey']]
                else:
                    xlims=None
                #ax.set_xlim(xlims)
                
                #===============================================================
                # loop and plot bounds
                #===============================================================
                for boundTag, col in err_df.items():
                    ax.hist(col.values, 
                            color=newColor_d[boundTag], alpha=0.3, 
                            label=boundTag, 
                            density=False, bins=40, 
                            range=xlims,
                            )
                    
                    if len(gdx2.columns.get_level_values(1))==1:
                        break #dont bother plotting bounds
                    
                    
                #===============================================================
                # #label
                #===============================================================
                # get float labels
                meta_d = {'cnt':len(err_df), 'zero_cnt':bx.all(axis=1).sum(), 'drop_zeros':drop_zeros,
                          'min':err_df.min().min(), 'max':err_df.max().max()}
 
                ax.text(0.4, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black')
                    
                
                #===============================================================
                # styling
                #===============================================================   
                ax.set_xlabel(row_key)                 
                # first columns
                if col_key == col_keys[0]:
                    """not setting for some reason"""
                    ax.set_ylabel('count')
 
                # last row
                if row_key == row_keys[-1]:
                    ax.legend()
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        
        return self.output_fig(fig,
                               out_dir = os.path.join(self.out_dir, tag), 
                               fname='model_smry_%03d_%s' %(modelID, self.longname))
        
 
    """
    plt.show()
    """
 
 
        
        
    def plot_total_bars(self, #generic total bar charts
                        
                    #data
                    dkey_d = {'rsamps':'mean','tvals':'var','tloss':'sum'}, #{dkey:groupby operation}
                    dx_raw=None,
                    modelID_l = None, #optinal sorting list
                    
                    #plot config
                    plot_rown='dkey',
                    plot_coln='studyArea',
                    plot_bgrp='modelID',
                    plot_colr=None,
                    sharey='row',
                    
                    #errorbars
                    qhi=0.99, qlo=0.01,
                    
                    #labelling
                    add_label=True,
                    baseline_loc='first_bar', #what to consider the baseline for labelling deltas
 
                    
                    #plot style
                    colorMap=None,
                    #ylabel=None,
                    
                    ):
        """"
        compressing a range of values
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_total_bars')
        if plot_colr is None: plot_colr=plot_bgrp
        
        if dx_raw is None: dx_raw = self.retrieve('outs')
 
 
        """
        view(dx)
        dx.loc[idx[0, 'LMFRA', 'LMFRA_0500yr', :], idx['tvals', 0]].sum()
        dx_raw.columns.unique('dkey')
        """
        
        
        log.info('on %s'%str(dx_raw.shape))
        
        #=======================================================================
        # data prep
        #=======================================================================
        #add requested indexers
        meta_indexers = set([plot_rown, plot_coln, plot_colr, plot_bgrp]).difference(['dkey']) #all those except dkey
        
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[list(dkey_d.keys()), :]], 
                                meta_indexers = meta_indexers,
                                modelID_l=modelID_l)
        
        mdex = dx.index
        """no... want to report stats per dkey group
        #move dkeys to index for consistency
        dx.stack(level=0)"""
        
        #get label dict
        lserx =  mdex.to_frame().reset_index(drop=True).loc[:, ['modelID', 'tag']
                           ].drop_duplicates().set_index('modelID').iloc[:,0]
                            
        mid_tag_d = {k:'%s (%s)'%(v, k) for k,v in lserx.items()}
        
        if modelID_l is None:
            modelID_l = mdex.unique('modelID').tolist()
        else:
            miss_l = set(modelID_l).difference( mdex.unique('modelID'))
            assert len(miss_l)==0, 'requested %i modelIDs not in teh data \n    %s'%(len(miss_l), miss_l)
            
        #=======================================================================
        # configure dimensions
        #=======================================================================
        if plot_rown=='dkey':
            row_keys = list(dkey_d.keys())
            axis=1
        else:
            dkey = list(dkey_d.keys())[0]
            axis=0
            row_keys = mdex.unique(plot_rown).tolist()
        
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
        """
        view(dx)
        plt.show()
        """
        col_keys = mdex.unique(plot_coln).tolist()
        
        
        
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,  # col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, sharex='all',  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        #fig.suptitle('%s total on %i studyAreas (%s)' % (lossType.upper(), len(mdex.unique('studyArea')), self.tag))
        
        #=======================================================================
        # #get colors
        #=======================================================================
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        if plot_colr == 'dkey_range':
            ckeys = ['hi', 'low', 'mean']
        else:
            ckeys = mdex.unique(plot_colr) 
        
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #===================================================================
        # loop and plot
        #===================================================================

        
        for col_key, gdx1 in dx.groupby(level=[plot_coln]):
            keys_d = {plot_coln:col_key}
            

            
            
            for row_key, gdx2 in gdx1.groupby(level=[plot_rown], axis=axis):
                keys_d[plot_rown] = row_key
                ax = ax_d[row_key][col_key]
                
                if plot_rown=='dkey':
                    dkey = row_key
 
                #===============================================================
                # data prep
                #===============================================================
 
                
                f = getattr(gdx2.groupby(level=[plot_bgrp]), dkey_d[dkey])
                
                try:
                    gdx3 = f()#.loc[modelID_l, :] #collapse assets, sort
                except Exception as e:
                    raise Error('failed on %s w/ \n    %s'%(keys_d, e))
                
                gb = gdx3.groupby(level=0, axis=1)   #collapse iters (gb object)
                
                #===============================================================
                #plot bars------
                #===============================================================
                #===============================================================
                # data setup
                #===============================================================
 
                barHeight_ser = gb.mean() #collapse iters(
                ylocs = barHeight_ser.T.values[0]
                
                #===============================================================
                # #formatters.
                #===============================================================
 
                # labels conversion to tag
                if plot_bgrp=='modelID':
                    tick_label = [mid_tag_d[mid] for mid in barHeight_ser.index] #label by tag
                else:
                    tick_label = ['%s=%s'%(plot_bgrp, i) for i in barHeight_ser.index]
                #tick_label = ['m%i' % i for i in range(0, len(barHeight_ser))]
  
                # widths
                bar_cnt = len(barHeight_ser)
                width = 0.9 / float(bar_cnt)
                
                #===============================================================
                # #add bars
                #===============================================================
                xlocs = np.linspace(0, 1, num=len(barHeight_ser))# + width * i
                bars = ax.bar(
                    xlocs,  # xlocation of bars
                    ylocs,  # heights
                    width=width,
                    align='center',
                    color=color_d.values(),
                    #label='%s=%s' % (plot_colr, ckey),
                    #alpha=0.5,
                    tick_label=tick_label,
                    )
                
                #===============================================================
                # add error bars--------
                #===============================================================
                if len(gdx2.columns.get_level_values(1))>1:
                    
                    #get error values
                    err_df = pd.concat({'hi':gb.quantile(q=qhi),'low':gb.quantile(q=qlo)}, axis=1).droplevel(axis=1, level=1)
                    
                    #convert to deltas
                    assert np.array_equal(err_df.index, barHeight_ser.index)
                    errH_df = err_df.subtract(barHeight_ser.values, axis=0).abs().T.loc[['low', 'hi'], :]
                    
                    #add the error bars
                    ax.errorbar(xlocs, ylocs,
                                errH_df.values,  
                                capsize=5, color='black',
                                fmt='none', #no data lines
                                )
                    """
                    plt.show()
                    """
                    
                #===============================================================
                # add labels--------
                #===============================================================
                if add_label:
                    log.debug(keys_d)
 
                    if dkey_d[dkey] == 'var':continue
                    #===========================================================
                    # #calc errors
                    #===========================================================
                    d = {'pred':barHeight_ser.T.values[0]}
                    
                    # get trues
                    if baseline_loc == 'first_bar':
                        d['true'] = np.full(len(barHeight_ser),d['pred'][0])
                    elif baseline_loc == 'first_axis':
                        if col_key == col_keys[0] and row_key == row_keys[0]:
                            base_ar = d['pred'].copy()
 
                        
                        d['true'] = base_ar
                    else:
                        raise Error('bad key')
                        
                    
                    d['delta'] = (d['pred'] - d['true']).round(3)
                    
                    # collect
                    tl_df = pd.DataFrame.from_dict(d)
                    
                    tl_df['relErr'] = (tl_df['delta'] / tl_df['true'])
                
                    tl_df['xloc'] = xlocs
                    #===========================================================
                    # add as labels
                    #===========================================================
                    for event, row in tl_df.iterrows():
                        ax.text(row['xloc'], row['pred'] * 1.01, #shifted locations
                                '%+.1f %%' % (row['relErr'] * 100),
                                ha='center', va='bottom', rotation='vertical',
                                fontsize=10, color='red')
                        
                    log.debug('added error labels \n%s' % tl_df)
                    
                    #expand the limits
                    ylims = ax.get_ylim()
                    
                    ax.set_ylim(tuple([1.1*x for x in ylims]))
                    
                #===============================================================
                # #wrap format subplot
                #===============================================================
                """
                fig.show()
                """
 
                ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
                # first row
                #===============================================================
                # if row_key == mdex.unique(plot_rown)[0]:
                #     pass
                #===============================================================
         
                        
                # first col
                if col_key == mdex.unique(plot_coln)[0]:
                    ylabel = '%s (%s)'%(dkey,  dkey_d[dkey])
                    ax.set_ylabel(ylabel)
                    
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        
        return self.output_fig(fig, fname='total_bars_%s' % (self.longname))
    
    


    def plot_dkey_mat(self, #flexible plotting of model results in a matrix
                  
                    #data
                    dkey='tvals', #column group w/ values to plot
                    dx_raw=None,
                    modelID_l = None, #optinal sorting list
                    
                    #plot config
                    plot_type='hist',
                    plot_rown='aggLevel',
                    plot_coln='dscale_meth',
                    plot_colr='dkey_range',
                    #plot_bgrp='modelID',
                    
                    #data control
                    xlims = None,
                    qhi=0.99, qlo=0.01,
                    drop_zeros=True,
                    
                    #labelling
                    add_label=True,
 
                    
                    #plot style
                    colorMap=None,
                    #ylabel=None,
                    sharey='all', sharex='col',
                    
                    #histwargs
                    bins=20, rwidth=0.9, 
                    mean_line=True, #plot a vertical line on the mean
                    fmt='svg',
                    ):
        """"
        generally 1 modelId per panel
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_dkey_mat')
        
        
        idn = self.idn
 
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
        
        
        log.info('on \'%s\' (%s x %s)'%(dkey, plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================

        
        #add requested indexers
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[dkey, :]], 
                                meta_indexers = set([plot_rown, plot_coln]),
                                modelID_l=modelID_l)
        
        log.info('on %s'%str(dx_raw.shape))
        mdex = dx.index
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, sharex=sharex,  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        fig.suptitle('\'%s\' values'%dkey)
        
        #=======================================================================
        # #get colors
        #=======================================================================
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        if plot_colr == 'dkey_range':
            ckeys = ['hi', 'low', 'mean']
        else:
            ckeys = mdex.unique(plot_colr) 
        
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        for gkeys, gdx0 in dx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
            #xlims = (gdx0.min().min(), gdx0.max().max())
            #===================================================================
            # prep data
            #===================================================================
                    
            # #drop empty iters
            bxcol = gdx0.isna().all(axis=0)
            
            if bxcol.any():
     
                log.warning('got %i/%i empty iters....dropping'%(bxcol.sum(), len(bxcol)))
                gdx0 = gdx0.loc[:,~bxcol]
            else:
                gdx0 = gdx0
            
            data_d, bx = self.prep_ranges(qhi, qlo, drop_zeros, gdx0)
            
            """
            view(gdx0)
            gdx0.index.droplevel('gid').to_frame().drop_duplicates().reset_index(drop=True)
            """
                
            #===================================================================
            # #get color
            #===================================================================
            if plot_colr == 'dkey_range':
                color = [color_d[k] for k in data_d.keys()]
            else:
                #all the same color
                color = [color_d[keys_d[plot_colr]] for k in data_d.keys()]
 
            #===================================================================
            # histogram of means
            #===================================================================
            if plot_type == 'hist':
 
                ar, _, patches = ax.hist(
                    data_d.values(),
                        range=xlims,
                        bins=bins,
                        density=False,  color = color, rwidth=rwidth,
                        label=list(data_d.keys()))
                
                bin_max = ar.max()
                #vertical mean line
                if mean_line:
                    ax.axvline(gdx0.mean().mean(), color='black', linestyle='dashed')
            #===================================================================
            # box plots
            #===================================================================
            elif plot_type=='box':
                #===============================================================
                # zero line
                #===============================================================
                #ax.axhline(0, color='black')
 
                #===============================================================
                # #add bars
                #===============================================================
                boxres_d = ax.boxplot(data_d.values(), labels=data_d.keys(), meanline=True,
                           # boxprops={'color':newColor_d[rowVal]}, 
                           # whiskerprops={'color':newColor_d[rowVal]},
                           # flierprops={'markeredgecolor':newColor_d[rowVal], 'markersize':3,'alpha':0.5},
                            )
 
                #===============================================================
                # add extra text
                #===============================================================
                
                # counts on median bar
                for gval, line in dict(zip(data_d.keys(), boxres_d['medians'])).items():
                    x_ar, y_ar = line.get_data()
                    ax.text(x_ar.mean(), y_ar.mean(), 'n%i' % len(data_d[gval]),
                            # transform=ax.transAxes, 
                            va='bottom', ha='center', fontsize=8)
                    
            #===================================================================
            # violin plot-----
            #===================================================================
            elif plot_type=='violin':

                #===============================================================
                # plot
                #===============================================================
                parts_d = ax.violinplot(data_d.values(),  
                                       showmeans=True,
                                       showextrema=True,  
                                       )
 
                #===============================================================
                # color
                #===============================================================
                #===============================================================
                # if len(data_d)>1:
                #     """nasty workaround for labelling"""                    
                #===============================================================
                labels = list(data_d.keys())
                #ckey_d = {i:color_key for i,color_key in enumerate(labels)}
 
 
                
                #style fills
                for i, pc in enumerate(parts_d['bodies']):
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(0.5)
                    
                #style lines
                for partName in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
                    parts_d[partName].set(color='black', alpha=0.5)
            else:
                raise Error(plot_type)
            
            if not xlims is None:
                ax.set_xlim(xlims)
            #===================================================================
            # labels
            #===================================================================
            if add_label:
                # get float labels
                meta_d = {'modelIDs':str(gdx0.index.unique(idn).tolist()),
                           'cnt':len(gdx0), 'zero_cnt':bx.sum(), 'drop_zeros':drop_zeros,
                           'iters':len(gdx0.columns),
                           'min':gdx0.min().min(), 'max':gdx0.max().max(), 'mean':gdx0.mean().mean()}
                
                if plot_type == 'hist':
                    meta_d['bin_max'] = bin_max
 
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black')
            
            #===================================================================
            # wrap format
            #===================================================================
            ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
            
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
 
                
                # first row
                if row_key == row_keys[0]:
                    #first col
                    if col_key == col_keys[0]:
                        if plot_type=='hist':
                            ax.legend()
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type == 'hist':
                        ax.set_ylabel('count')
                    elif plot_type in ['box', 'violin']:
                        ax.set_ylabel(dkey)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type == 'hist':
                        ax.set_xlabel(dkey)
                    elif plot_type=='violin':
                        
                        ax.set_xticks(np.arange(1, len(labels) + 1))
                        ax.set_xticklabels(labels)
                    #last col
                    if col_key == col_keys[-1]:
                        pass
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        """
        plt.show()
        """
        
        return self.output_fig(fig, fname='%s_%s_%sx%s_%s' % (dkey, plot_type, plot_rown, plot_coln, self.longname), fmt=fmt)
            
    def plot_compare_mat(self, #flexible plotting of model results vs. true in a matrix
                  
                    #data
                    dkey='tvals',#column group w/ values to plot
                    aggMethod='mean', #method to use for aggregating the true values (down to the gridded)
                    true_dx_raw=None, #base values (indexed to raws per model)
                    baseID=0,
                    dx_raw=None, #combined model results
                    modelID_l = None, #optinal sorting list
                    
                    #plot config
                    plot_type='scatter', 
                    plot_rown='aggLevel',
                    plot_coln='resolution',
                    plot_colr=None,
                    
                    #plot config [bars]
                    plot_bgrp=None, #grouping (for plotType==bars)
                    err_type='absolute', #what type of errors to calculate (for plot_type='bars')
                        #absolute: modelled - true
                        #relative: absolute/true
 
                    
                    #data control
                    xlims = None,
                    qhi=0.99, qlo=0.01,
                    #drop_zeros=True, #must always be false for the matching to work
                    
                    #labelling
                    #baseID=None, 
                    add_label=True,
                    title=None,
 
                    
                    #plot style
                    
                    colorMap=None,
                    sharey=None,sharex=None,
                    **kwargs):
        """"
        generally 1 modelId per panel
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_compare_mat')
        
        if plot_colr is None: 
            plot_colr=plot_bgrp
        
        if plot_colr is None: 
            plot_colr=plot_rown
            
        if plot_bgrp is None:

            plot_bgrp = plot_colr
            
        idn = self.idn
        #if baseID is None: baseID=self.baseID
 
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
            
        if true_dx_raw is None:
            true_d = self.retrieve('trues')
            true_dx_raw = true_d[baseID]
            
        if sharey is None:
            if plot_type=='scatter':
                sharey='none'
            else:
                sharey='all'
                
        if sharex is None:
            if plot_type=='scatter':
                sharex='none'
            else:
                sharex='all'
                
        if plot_type=='bars':
            #assert err_type in ['absolute', 'relative']
            assert isinstance(plot_bgrp, str)
        
        
        log.info('on \'%s\' (%s x %s)'%(dkey, plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================
        assert_func(lambda: self.check_mindex_match(true_dx_raw.index, dx_raw.index), msg='raw vs trues')
        
        #add requested indexers
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[dkey, :]], 
                                meta_indexers = set([plot_rown, plot_coln]),
                                modelID_l=modelID_l)
        
        log.info('on %s'%str(dx.shape))
        mdex = dx.index
        
        #and on the trues
        true_dx = self.join_meta_indexers(dx_raw = true_dx_raw.loc[:, idx[dkey, :]], 
                                meta_indexers = set([plot_rown, plot_coln]),
                                modelID_l=modelID_l)
        
        
 

 
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, 
                                    sharex=sharex,  
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        
        #=======================================================================
        # title
        #=======================================================================
        if title is None:
            if not plot_type=='bars':
                title = '\'%s\' errors'%dkey
            else:
                title = '\'%s\' errors (%s)'%(dkey, err_type)
            
        fig.suptitle(title)
        #=======================================================================
        # #get colors
        #=======================================================================
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
 
        ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
 
 
        true_gb = true_dx.groupby(level=[plot_coln, plot_rown])
        for gkeys, gdx0 in dx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)

            #===================================================================
            # data prep----------
            #===================================================================
            gdx1 = gdx0
 
            #get the corresponding true values
            tgdx0 = true_gb.get_group(gkeys)
            
            #===================================================================
            # aggregate the trues
            #===================================================================
            """because trues are mapped from the base model.. here we compress down to the model index"""
            tgdx1 = getattr(tgdx0.groupby(level=[gdx1.index.names]), aggMethod)()
            tgdx2 = tgdx1.reorder_levels(gdx1.index.names).sort_index()
            
            if not np.array_equal(gdx1.index, tgdx2.index):
                
                miss_l = set(gdx1.index.unique(5)).symmetric_difference(tgdx2.index.unique(5))
                #assert_index_equal(gdx1.index, tgdx2.index)
                
                """
                bx = gdx1.index.get_level_values(5).isin(miss_l)
                view(gdx1.loc[bx, :])
                
                """
                
                if len(miss_l)>0:
                    raise Error('%i/%i true keys dont match on %s \n    %s'%(len(miss_l), len(gdx1), keys_d, miss_l))
                else:
                    raise Error('bad indexers on %s modelIDs:\n    %s'%(keys_d, tgdx2.index.unique('modelID').tolist()))
            
            
            
            #===================================================================
            # meta
            #===================================================================
            meta_d = { 'modelIDs':str(list(gdx0.index.unique(idn))),
                            'drop_zeros':False,'iters':len(gdx1.columns),
                            }
                            
            
            #===================================================================
            # scatter plot-----
            #===================================================================
            """consider hist2d?"""
            if plot_type =='scatter':
                #===================================================================
                # reduce ranges
                #===================================================================
                #model results
                data_d, zeros_bx = self.prep_ranges(qhi, qlo, False, gdx1)
                
                #trues
                true_data_d, _ = self.prep_ranges(qhi, qlo, False, tgdx2)
            
                """only using mean values for now"""
                xar, yar = data_d['mean'], true_data_d['mean'] 
                
                stat_d = self.ax_corr_scat(ax, xar, yar, 
                                           #label='%s=%s'%(plot_colr, keys_d[plot_colr]),
                                           scatter_kwargs = {
                                               'color':color_d[keys_d[plot_colr]]
                                               },
                                           logger=log, add_label=False)
                
                gdata = gdx1 #for stats
                #===============================================================
                # meta
                #===============================================================
                meta_d.update(stat_d)
                
                #add confusion matrix stats
                cm_df, cm_dx = self.get_confusion(pd.DataFrame({'pred':xar, 'true':yar}), logger=log)
                
                meta_d.update(cm_dx.droplevel(['pred', 'true']).iloc[:,0].to_dict())
                
            #===================================================================
            # bar plot---------
            #===================================================================
            elif plot_type=='bars':
                """TODO: consolidate w/ plot_total_bars
                integrate with write_suite_smry errors"""
                #===============================================================
                # data setup
                #===============================================================
                gdx2 = gdx1 - tgdx2 #modelled - trues
                
                gb = gdx2.groupby(level=plot_bgrp)
                
                
                barHeight_ser = gb.sum() #collapse iters(
                
                
                
                if err_type=='relative':
                    barHeight_ser = barHeight_ser/tgdx2.groupby(level=plot_bgrp).sum()
                elif err_type=='bias':
                    predTotal_ser= gdx1.groupby(level=plot_bgrp).sum()
                    trueTotal_ser=tgdx2.groupby(level=plot_bgrp).sum()
                    barHeight_ser = predTotal_ser/trueTotal_ser
                elif err_type=='absolute':
                    pass
                else:
                    raise IOError('bad err_type: %s'%err_type)
 
                    
                """always want totals for the bars"""
                
                ylocs = barHeight_ser.T.values[0]
                gdata = gdx2
                
                #===============================================================
                # #formatters.
                #===============================================================
 
                # labels conversion to tag
                if plot_bgrp=='modelID':
                    raise Error('not implementd')
                    #tick_label = [mid_tag_d[mid] for mid in barHeight_ser.index] #label by tag
                else:
                    tick_label = ['%s=%s'%(plot_bgrp, i) for i in barHeight_ser.index]
                #tick_label = ['m%i' % i for i in range(0, len(barHeight_ser))]
  
                # widths
                bar_cnt = len(barHeight_ser)
                width = 0.9 / float(bar_cnt)
                
                #===============================================================
                # #add bars
                #===============================================================
                xlocs = np.linspace(0, 1, num=len(barHeight_ser))# + width * i
                bars = ax.bar(
                    xlocs,  # xlocation of bars
                    ylocs,  # heights
                    width=width,
                    align='center',
                    color=color_d.values(),
                    #label='%s=%s' % (plot_colr, ckey),
                    #alpha=0.5,
                    tick_label=tick_label,
                    )
                
                #===============================================================
                # add error bars 
                #===============================================================
                if len(gdx2.columns.get_level_values(1))>1:
                    """untesetd"""
                    #get error values
                    err_df = pd.concat({'hi':gb.quantile(q=qhi),'low':gb.quantile(q=qlo)}, axis=1).droplevel(axis=1, level=1)
                    
                    #convert to deltas
                    assert np.array_equal(err_df.index, barHeight_ser.index)
                    errH_df = err_df.subtract(barHeight_ser.values, axis=0).abs().T.loc[['low', 'hi'], :]
                    
                    #add the error bars
                    ax.errorbar(xlocs, ylocs,
                                errH_df.values,  
                                capsize=5, color='black',
                                fmt='none', #no data lines
                                )
                
                #===============================================================
                # add bar labels
                #===============================================================
                d1 = {k:pd.Series(v, dtype=float) for k,v in {'yloc':ylocs, 'xloc':xlocs}.items()}

                for event, row in pd.concat(d1, axis=1).iterrows():
                    if err_type=='bias':
                        txt = '%.2f' %(row['yloc'])
                    else:
                        txt = '%+.1f %%' % (row['yloc'] * 100)
                    ax.text(row['xloc'], row['yloc'] * 1.01, #shifted locations
                                txt,ha='center', va='bottom', rotation='vertical',fontsize=10, color='red')
                    
            #===================================================================
            # violin plot-----
            #===================================================================
            elif plot_type=='violin':
                #===============================================================
                # data setup
                #===============================================================
                gdx2 = gdx1 - tgdx2 #modelled - trues
                
                gb = gdx2.groupby(level=plot_bgrp)
                
                data_d = {k:v.values.T[0] for k,v in gb}
                gdata = gdx2
                #===============================================================
                # plot
                #===============================================================
                parts_d = ax.violinplot(data_d.values(),  
 
                                       showmeans=True,
                                       showextrema=True,  
                                       )
                
                #===============================================================
                # color
                #===============================================================
                labels = list(data_d.keys())
                ckey_d = {i:color_key for i,color_key in enumerate(labels)}
                
                #style fills
                for i, pc in enumerate(parts_d['bodies']):
                    pc.set_facecolor(color_d[ckey_d[i]])
                    pc.set_edgecolor(color_d[ckey_d[i]])
                    pc.set_alpha(0.5)
                    
                #style lines
                for partName in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
                    parts_d[partName].set(color='black', alpha=0.5)
                
                
                
                
 
            
            else:
                raise KeyError('unrecognized plot_type: %s'%plot_type)
            """
            fig.show()
            """
            #===================================================================
            # post-format----
            #===================================================================
            ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
            #===================================================================
            # labels
            #===================================================================
            if add_label:
                # get float labels
                meta_d.update({ 
                    'min':gdata.min().min(), 'max':gdata.max().max(), 'mean':gdata.mean().mean(),
                          })
 
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black')
                
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                if plot_type=='scatter':
                    ax.legend(loc=1)
                # first row
                if row_key == row_keys[0]:
                    pass
                #last col
                if col_key == col_keys[-1]:
                    pass
                    
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type in ['bars']:
                        ax.set_ylabel('\'%s\' total errors (%s)'%(dkey, err_type))
                    elif plot_type == 'violin':
                        ax.set_ylabel('\'%s\' errors'%(dkey))
                    elif plot_type=='scatter':
                        ax.set_ylabel('\'%s\' (true)'%dkey)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type == 'bars': 
                        pass
                        #ax.set_ylabel('\'%s\' (agg - true)'%dkey)
                    elif plot_type=='violin':
                        ax.set_xticks(np.arange(1, len(labels) + 1))
                        ax.set_xticklabels(labels)
                        
                    else:
                        ax.set_xlabel('\'%s\' (aggregated)'%dkey)
                        
                    
 
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        """
        plt.show()
        """
        
        return self.output_fig(fig, fname='compareMat_%s_%s_%sX%s_%s' % (
            title.replace(' ','').replace('\'',''),
             plot_type, plot_rown, plot_coln, self.longname), **kwargs)
 
    def plot_vs_mat(self, #plot dkeys against eachother in a matrix
                  
                    #data
                    dkey_y='rloss',#column group w/ values to plot
                    dkey_x='rsamps',
                     
 
                    dx_raw=None, #combined model results
                    modelID_l = None, #optinal sorting list
                    
                    #plot config
                    #plot_type='hist',
                    plot_rown='studyArea',
                    plot_coln='vid',
                    plot_colr=None,
                    #plot_bgrp='modelID',
                    
                    #data control
                    #xlims = None,
                    qhi=0.99, qlo=0.01,
                    #drop_zeros=True, #must always be false for the matching to work
                    
                    #labelling
                    baseID=None,
                    add_label=True,
 
                    
                    #plot style
                    colorMap=None,
                    **kwargs):
        """"
        generally 1 modelId per panel
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_compare_mat')
        
        if plot_colr is None: plot_colr=plot_rown
        idn = self.idn
        #if baseID is None: baseID=self.baseID
 
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
            
 
        
        
        log.info('on \'%s\' vs \'%s\' (%s x %s)'%(dkey_x, dkey_y, plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================
        #add requested indexers
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[[dkey_x, dkey_y], :]], 
                                meta_indexers = set([plot_rown, plot_coln]),
                                modelID_l=modelID_l)
        
        log.info('on %s'%str(dx.shape))
        mdex = dx.index
        
 
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='all', sharex='all',  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        
        fig.suptitle('\'%s\' vs \'%s\' '%(dkey_x, dkey_y))
        
        #=======================================================================
        # #get colors
        #=======================================================================
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
 
        ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
 
        for gkeys, gdx0 in dx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
            #===================================================================
            # color loop
            #===================================================================
            for color_key, gdx1 in gdx0.groupby(level=plot_colr):
            
                #===================================================================
                # data prep----------
                #==================================================================
                #split the data
                xdx0 = gdx1.loc[:, idx[dkey_x, :]].droplevel(0, axis=1)
                ydx0 = gdx1.loc[:, idx[dkey_y, :]].droplevel(0, axis=1)
     
                
                #===================================================================
                # ranges
                #===================================================================
                #model results
                xdata_d, zeros_bx = self.prep_ranges(qhi, qlo, False, xdx0)
                
                #trues
                ydata_d, _ = self.prep_ranges(qhi, qlo, False, ydx0)
                
     
                
                #===================================================================
                # scatter plot
                #===================================================================
                """only using mean values for now"""
                xar, yar = xdata_d['mean'], ydata_d['mean'] 
                
      
                ax.plot(xar, yar, linestyle='None', color=color_d[color_key],
                        label=color_key,
                         **{'markersize':3.0, 'marker':'.', 'fillstyle':'full'})
            """
            fig.show()
            """
            #===================================================================
            # post-format----
            #===================================================================
            ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
            #===================================================================
            # labels
            #===================================================================
            if add_label:
                # get float labels
                meta_d = {'modelIDs':str(list(gdx0.index.unique(idn))),
                           'count':len(gdx0), 'zero_cnt':(gdx0==0).sum().sum(), 'drop_zeros':False,
                           'iters':len(xdx0.columns),
                           #'min':gdx1.min().min(), 'max':gdx1.max().max(), 'mean':gdx1.mean().mean()},
                            }
 
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black')
                
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
 
                ax.legend(loc=1)
                # first row
                if row_key == row_keys[0]:
                    pass
                #last col
                if col_key == col_keys[-1]:
                    pass
                    
                
                        
                # first col
                if col_key == col_keys[0]:
                    ax.set_ylabel('\'%s\''%dkey_y)
                
                #last row
                if row_key == row_keys[-1]:
                    ax.set_xlabel('\'%s\''%dkey_x)
 
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        """
        plt.show()
        """
        
        return self.output_fig(fig, fname='%s-%s_%sx%s_%s' % (dkey_x, dkey_y, plot_rown, plot_coln, self.longname), **kwargs)
        

        
    #===========================================================================
    # OLD PLOTTER-------
    #===========================================================================
    def xxxplot_depths(self,
                    # data control
                    plot_fign='studyArea',
                    plot_rown='grid_size',
                    plot_coln='event',
                    plot_zeros=False,
                    serx=None,
                    
                    # style control
                    xlims=(0, 2),
                    ylims=(0, 2.5),
                    calc_str='points',
                    
                    out_dir=None,
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_depths')
        if out_dir is None: out_dir = self.out_dir
        if serx is None: serx = self.retrieve('rsamps')
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        
        assert serx.notna().all().all(), 'drys should be zeros'

        """
        plt.show()
        self.retrieve('tvals')
        view(serx)
        """
        #=======================================================================
        # loop on studyAreas
        #=======================================================================
        
        log.info('on %i' % len(serx))
        
        res_d = dict()
        for i, (sName, gsx1r) in enumerate(serx.groupby(level=plot_fign)):
            plt.close('all')
            gsx1 = gsx1r.droplevel(plot_fign)
            mdex = gsx1.index
            
            fig, ax_d = self.get_matrix_fig(
                                    gsx1.index.unique(plot_rown).tolist(),  # row keys
                                    gsx1.index.unique(plot_coln).tolist(),  # col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='all', sharex='all',  # everything should b euniform
                                    fig_id=i,
                                    set_ax_title=True,
                                    )
            
            for (row_key, col_key), gsx2r in gsx1r.groupby(level=[plot_rown, plot_coln]):
                #===============================================================
                # #prep data
                #===============================================================
                gsx2 = gsx2r.droplevel([plot_rown, plot_coln, plot_fign])
                
                if plot_zeros:
                    ar = gsx2.values
                else:
                    bx = gsx2 > 0.0
                    ar = gsx2[bx].values
                
                if not len(ar) > 0:
                    log.warning('no values for %s.%s.%s' % (sName, row_key, col_key))
                    continue
                #===============================================================
                # #plot
                #===============================================================
                ax = ax_d[row_key][col_key]
                ax.hist(ar, color='blue', alpha=0.3, label=row_key, density=True, bins=30, range=xlims)
                
                #===============================================================
                # #label
                #===============================================================
                # get float labels
                meta_d = {'calc_method':calc_str, plot_rown:row_key, 'wet':len(ar), 'dry':(gsx2 <= 0.0).sum(),
                           'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
 
                ax.text(0.5, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='blue')

                #===============================================================
                # styling
                #===============================================================                    
                # first columns
                if col_key == mdex.unique(plot_coln)[0]:
                    """not setting for some reason"""
                    ax.set_ylabel('density')
 
                # first row
                if row_key == mdex.unique(plot_rown)[0]:
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)
                    pass
                    # ax.set_title('event \'%s\''%(rlayName))
                    
                # last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel('depth (m)')
                    
            fig.suptitle('depths for studyArea \'%s\' (%s)' % (sName, self.tag))
            #===================================================================
            # wrap figure
            #===================================================================
            res_d[sName] = self.output_fig(fig, out_dir=os.path.join(out_dir, sName), fname='depths_%s_%s' % (sName, self.longname))

        #=======================================================================
        # warp
        #=======================================================================
        log.info('finished writing %i figures' % len(res_d))
        
        return res_d

    def xxxplot_tvals(self,
                    plot_fign='studyArea',
                    plot_rown='grid_size',
                    # plot_coln = 'event',
                    
                    plot_zeros=True,
                    xlims=(0, 200),
                    ylims=None,
                    
                    out_dir=None,
                    color='orange',
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_tvals')
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
 
        serx = self.retrieve('tvals')
        
        assert serx.notna().all().all(), 'drys should be zeros'

        """
        self.retrieve('tvals')
        view(serx)
        """
        #=======================================================================
        # loop on studyAreas
        #=======================================================================
        
        log.info('on %i' % len(serx))
        
        col_key = ''
        res_d = dict()
        for i, (sName, gsx1r) in enumerate(serx.groupby(level=plot_fign)):
            plt.close('all')
            gsx1 = gsx1r.droplevel(plot_fign)
            mdex = gsx1.index
            
            fig, ax_d = self.get_matrix_fig(
                                    gsx1.index.unique(plot_rown).tolist(),  # row keys
                                    [col_key],  # col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='all', sharex='all',  # everything should b euniform
                                    fig_id=i,
                                    set_ax_title=True,
                                    )
            
            for row_key, gsx2r in gsx1r.groupby(level=plot_rown):
                #===============================================================
                # #prep data
                #===============================================================
                gsx2 = gsx2r.droplevel([plot_rown, plot_fign])
                bx = gsx2 > 0.0
                if plot_zeros:
                    ar = gsx2.values
                else:
                    
                    ar = gsx2[bx].values
                
                if not len(ar) > 0:
                    log.warning('no values for %s.%s.%s' % (sName, row_key,))
                    continue
                #===============================================================
                # #plot
                #===============================================================
                ax = ax_d[row_key][col_key]
                ax.hist(ar, color=color, alpha=0.3, label=row_key, density=True, bins=30, range=xlims)
                
                # label
                meta_d = {
                    plot_rown:row_key,
                    'cnt':len(ar), 'zeros_cnt':np.invert(bx).sum(), 'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
                
                txt = '\n'.join(['%s=%.2f' % (k, v) for k, v in meta_d.items()])
                ax.text(0.5, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='blue')

                #===============================================================
                # styling
                #===============================================================                    
                # first columns
                #===============================================================
                # if col_key == mdex.unique(plot_coln)[0]:
                #     """not setting for some reason"""
                #===============================================================
                ax.set_ylabel('density')
 
                # first row
                if row_key == mdex.unique(plot_rown)[0]:
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)
                    pass
                    # ax.set_title('event \'%s\''%(rlayName))
                    
                # last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel('total value (scale)')
                    
            fig.suptitle('depths for studyArea \'%s\' (%s)' % (sName, self.tag))
            #===================================================================
            # wrap figure
            #===================================================================
            res_d[sName] = self.output_fig(fig, out_dir=os.path.join(out_dir, sName), fname='depths_%s_%s' % (sName, self.longname))

        #=======================================================================
        # warp
        #=======================================================================
        log.info('finished writing %i figures' % len(res_d))
        
        return res_d      


    def xxxplot_terrs_box(self,  # boxplot of total errors
                    
                    # data control
                    dkey='errs',
                    ycoln=('tl', 'delta'),  # values to plot
                    plot_fign='studyArea',
                    plot_rown='event',
                    plot_coln='vid',
                    plot_colr='grid_size',
                    # plot_bgrp = 'event',
                    
                    # plot style
                    ylabel=None,
                    colorMap=None,
                    add_text=True,
                    
                    out_dir=None,
                   ):
        """
        matrix figure
            figure: studyAreas
                rows: grid_size
                columns: events
                values: total loss sum
                colors: grid_size
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_terr_box')
        if colorMap is None: colorMap = self.colorMap
        if ylabel is None: ylabel = dkey
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        log.info('on \'%s\' w/ %i' % (dkey, len(dx_raw)))
        
        #=======================================================================
        # setup data
        #=======================================================================

        # make slice
        dxser = dx_raw.loc[:, ycoln]
 
        #=======================================================================
        # loop on figures
        #=======================================================================
        for i, (fig_key, gser0r) in enumerate(dxser.groupby(level=plot_fign)):
            
            mdex = gser0r.index
            plt.close('all')
            
            fig, ax_d = self.get_matrix_fig(
                                        mdex.unique(plot_rown).tolist(),  # row keys
                                        mdex.unique(plot_coln).tolist(),  # col keys
                                        figsize_scaler=4,
                                        constrained_layout=True,
                                        sharey='all',
                                        sharex='none',  # events should all be uniform
                                        fig_id=i,
                                        set_ax_title=True,
                                        )
            
            s = '-'.join(ycoln)
            fig.suptitle('%s for %s:%s (%s)' % (s, plot_fign, fig_key, self.tag))
 
            """
            fig.show()
            """
            
            #===================================================================
            # loop and plot
            #===================================================================
            for (row_key, col_key), gser1r in gser0r.droplevel(plot_fign).groupby(level=[plot_rown, plot_coln]):
                
                # data setup
                gser1 = gser1r.droplevel([plot_rown, plot_coln])
     
                # subplot setup 
                ax = ax_d[row_key][col_key]
                
                # group values
                gd = {k:g.values for k, g in gser1.groupby(level=plot_colr)}
                
                #===============================================================
                # zero line
                #===============================================================
                ax.axhline(0, color='red')
 
                #===============================================================
                # #add bars
                #===============================================================
                boxres_d = ax.boxplot(gd.values(), labels=gd.keys(), meanline=True,
                           # boxprops={'color':newColor_d[rowVal]}, 
                           # whiskerprops={'color':newColor_d[rowVal]},
                           # flierprops={'markeredgecolor':newColor_d[rowVal], 'markersize':3,'alpha':0.5},
                            )
                
                #===============================================================
                # add extra text
                #===============================================================
                
                # counts on median bar
                for gval, line in dict(zip(gd.keys(), boxres_d['medians'])).items():
                    x_ar, y_ar = line.get_data()
                    ax.text(x_ar.mean(), y_ar.mean(), 'n%i' % len(gd[gval]),
                            # transform=ax.transAxes, 
                            va='bottom', ha='center', fontsize=8)
                    
                    #===========================================================
                    # if add_text:
                    #     ax.text(x_ar.mean(), ylims[0]+1, 'mean=%.2f'%gd[gval].mean(), 
                    #         #transform=ax.transAxes, 
                    #         va='bottom',ha='center',fontsize=8, rotation=90)
                    #===========================================================
                    
                #===============================================================
                # #wrap format subplot
                #===============================================================
                ax.grid(alpha=0.8)
                # first row
                if row_key == mdex.unique(plot_rown)[0]:
                     
                    # last col
                    if col_key == mdex.unique(plot_coln)[-1]:
                        # ax.legend()
                        pass
                         
                # first col
                if col_key == mdex.unique(plot_coln)[0]:
                    ax.set_ylabel(ylabel)
                    
                # last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel(plot_colr)
            #===================================================================
            # wrap fig
            #===================================================================
            log.debug('finsihed %s' % fig_key)
            self.output_fig(fig, fname='box_%s_%s' % (s, self.longname),
                            out_dir=os.path.join(out_dir, fig_key))

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
 
    def xxxplot_errs_scatter(self,  # scatter plot of error-like data
                    # data control   
                    dkey='errs',
                    
                    # lossType='rl',
                    ycoln=('rl', 'delta'),
                    xcoln=('depth', 'grid'),
                       
                    # figure config
                    folder_varn='studyArea',
                    plot_fign='event',
                    plot_rown='grid_size',
                    plot_coln='vid',
                    plot_colr=None,
                    # plot_bgrp = 'event',
                    
                    plot_vf=False,  # plot the vf
                    plot_zeros=False,
                    
                    # axconfig
                    ylims=None,
                    xlims=None,
                    
                    # plot style
                    ylabel=None,
                    xlabel=None,
                    colorMap=None,
                    add_text=True,
                    
                    # outputs
                    fmt='png', transparent=False,
                    out_dir=None,
                   ):
        
        # raise Error('lets fit a regression to these results')
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_errs_scatter')
        if colorMap is None: colorMap = self.colorMap
        if ylabel is None: ylabel = '.'.join(ycoln)
        if xlabel is None: xlabel = '.'.join(xcoln)
        
        #=======================================================================
        # if plot_vf:
        #     assert lossType=='rl'
        #=======================================================================
            
        if plot_colr is None: plot_colr = plot_rown
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
 
        if plot_vf:
            vf_d = self.retrieve('vf_d')
        log.info('on \'%s\' for %s vs %s w/ %i' % (dkey, xcoln, ycoln, len(dx_raw)))
        
        #=======================================================================
        # prep data
        #=======================================================================
        # get slice specified by user
        dx1 = pd.concat([dx_raw.loc[:, ycoln], dx_raw.loc[:, xcoln]], axis=1)
        dx1.columns.set_names(dx_raw.columns.names, inplace=True)
 
        #=======================================================================
        # plotting setup
        #=======================================================================
        # get colors
        cvals = dx_raw.index.unique(plot_colr)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        """
        plt.show()
        """
        #=======================================================================
        # loop an study area/folders
        #=======================================================================
        
        for folder_key, gdx0 in dx1.groupby(level=folder_varn, axis=0):
 
            #=======================================================================
            # loop on figures
            #=======================================================================
            od = os.path.join(out_dir, folder_key, xlabel)
            plt.close('all')
        
            for i, (fig_key, gdx1) in enumerate(gdx0.groupby(level=plot_fign, axis=0)):
                keys_d = dict(zip([folder_varn, plot_fign], (folder_key, fig_key)))
                mdex = gdx1.index
                
                fig, ax_d = self.get_matrix_fig(
                                            mdex.unique(plot_rown).tolist(),  # row keys
                                            mdex.unique(plot_coln).tolist(),  # col keys
                                            figsize_scaler=4,
                                            constrained_layout=True,
                                            sharey='all',
                                            sharex='all',  # events should all be uniform
                                            fig_id=i,
                                            set_ax_title=False,
                                            )
                
                s = ' '.join(['%s:%s' % (k, v) for k, v in keys_d.items()])
                fig.suptitle('%s vs %s for %s' % (xcoln, ycoln, s))
            
                """
                fig.show()
                """
            
                #===================================================================
                # loop on axis row/column (and colors)----------
                #===================================================================
                for (row_key, col_key, ckey), gdx2 in gdx1.groupby(level=[plot_rown, plot_coln, plot_colr]):
                    keys_d.update(
                        dict(zip([plot_rown, plot_coln, plot_colr], (row_key, col_key, ckey)))
                        )
                    # skip trues
                    #===========================================================
                    # if ckey == 0:
                    #     continue 
                    #===========================================================
                    # subplot setup 
                    ax = ax_d[row_key][col_key]
 
                    #===============================================================
                    # prep data
                    #===============================================================
 
                    dry_bx = gdx2[xcoln] <= 0.0
                    
                    if not plot_zeros:
                        xar, yar = gdx2.loc[~dry_bx, xcoln].values, gdx2.loc[~dry_bx, ycoln].values
                    else:
                        xar, yar = gdx2[xcoln].values, gdx2[ycoln].values

                    #===============================================================
                    # zero line
                    #===============================================================
                    ax.axhline(0, color='black', alpha=0.8, linewidth=0.5)
                    
                    #===============================================================
                    # plot function
                    #===============================================================
                    if plot_vf:
                        vf_d[col_key].plot(ax=ax, logger=log, set_title=False,
                                           lineKwargs=dict(
                            color='black', linestyle='dashed', linewidth=1.0, alpha=0.9)) 
                    
                    #===============================================================
                    # #add scatter plot
                    #===============================================================
                    ax.plot(xar, yar,
                               color=newColor_d[ckey], markersize=4, marker='x', alpha=0.8,
                               linestyle='none',
                                   label='%s=%s' % (plot_colr, ckey))
 
                    #===========================================================
                    # add text
                    #===========================================================

                    if add_text:
                        meta_d = {'ycnt':len(yar),
                                  'dry_cnt':dry_bx.sum(),
                                  'wet_cnt':np.invert(dry_bx).sum(),
                                  'y0_cnt':(yar == 0.0).sum(),
                                  'ymean':yar.mean(), 'ymin':yar.min(), 'ymax':yar.max(),
                                  'xmax':xar.max(),
                              # 'plot_zeros':plot_zeros,
                              }
                        
                        if ycoln[1] == 'delta':
                            meta_d['rmse'] = ((yar ** 2).mean()) ** 0.5
                                            
                        txt = '\n'.join(['%s=%.2f' % (k, v) for k, v in meta_d.items()])
                        ax.text(0.1, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='black')
     
                    #===============================================================
                    # #wrap format subplot
                    #===============================================================
                    ax.set_title('%s=%s and %s=%s' % (
                         plot_rown, row_key, plot_coln, col_key))
                    
                    # first row
                    if row_key == mdex.unique(plot_rown)[0]:
                        
                        # last col
                        if col_key == mdex.unique(plot_coln)[-1]:
                            pass
                             
                    # first col
                    if col_key == mdex.unique(plot_coln)[0]:
                        ax.set_ylabel(ylabel)
                        
                    # last row
                    if row_key == mdex.unique(plot_rown)[-1]:
                        ax.set_xlabel(xlabel)
                        
                    # loast col
                    if col_key == mdex.unique(plot_coln)[-1]:
                        pass
                        # ax.legend()
                        
                #===================================================================
                # post format
                #===================================================================
                for row_key, ax0_d in ax_d.items():
                    for col_key, ax in ax0_d.items():
                        ax.grid()
                        
                        if not ylims is None:
                            ax.set_ylim(ylims)
                        
                        if not xlims is None:
                            ax.set_xlim(xlims)
                #===================================================================
                # wrap fig
                #===================================================================
                log.debug('finsihed %s' % fig_key)
                s = '_'.join(['%s' % (keys_d[k]) for k in [ folder_varn, plot_fign]])
                
                s2 = ''.join(ycoln) + '-' + ''.join(xcoln)
                
                self.output_fig(fig, out_dir=od,
                                fname='scatter_%s_%s_%s' % (s2, s, self.longname.replace('_', '')),
                                fmt=fmt, transparent=transparent, logger=log)
            """
            fig.show()
            """

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
    
    def xxxplot_accuracy_mat(self,  # matrix plot of accuracy
                    # data control   
                    dkey='errs',
                    lossType='tl',
                    
                    folder_varns=['studyArea', 'event'],
                    plot_fign='vid',  # one raster:vid per plot
                    plot_rown='grid_size',
                    plot_zeros=True,

                    # output control
                    out_dir=None,
                    fmt='png',
                    
                    # plot style
                    binWidth=None,
                    colorMap=None,
                    lims_d={'raw':{'x':None, 'y':None}}  # control limits by column
                    # add_text=True,
                   ):
        
        """
        row1: trues
        rowx: grid sizes
        
        col1: hist of raw 'grid' values (for this lossType)
        col2: hist of delta values
        col3: scatter of 'grid' vs. 'true' values 
            """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_accuracy_mat.%s' % lossType)
        if colorMap is None: colorMap = self.colorMap
 
        if out_dir is None: out_dir = self.out_dir
        col_keys = ['raw', 'delta', 'correlation']
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        
        # slice by user
        dxind1 = dx_raw.loc[:, idx[lossType, ['grid', 'true', 'delta']]].droplevel(0, axis=1)
        """
        dx_raw.columns
        view(dx_raw)
        """
        # get colors
        cvals = dx_raw.index.unique(plot_rown)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        #=======================================================================
        # helpers
        #=======================================================================
        lim_max_d = {'raw':{'x':(0, 0), 'y':(0, 0)}, 'delta':{'x':(0, 0), 'y':(0, 0)}}

        def upd_lims(key, ax):
            # x axis
            lefti, righti = ax.get_xlim()
            leftj, rightj = lim_max_d[key]['x'] 
            
            lim_max_d[key]['x'] = (min(lefti, leftj), max(righti, rightj))
            
            # yaxis
            lefti, righti = ax.get_ylim()
            leftj, rightj = lim_max_d[key]['y'] 
            
            lim_max_d[key]['y'] = (min(lefti, leftj), max(righti, rightj))
        
        def set_lims(key, ax):
            if key in lims_d:
                if 'x' in lims_d[key]:
                    ax.set_xlim(lims_d[key]['x'])
                if 'y' in lims_d[key]:
                    ax.set_ylim(lims_d[key]['y'])
            
            upd_lims(key, ax)
        #=======================================================================
        # loop and plot----------
        #=======================================================================
        
        log.info('for \'%s\' w/ %i' % (lossType, len(dxind1)))
        for fkeys, gdxind1 in dxind1.groupby(level=folder_varns):
            keys_d = dict(zip(folder_varns, fkeys))
            
            for fig_key, gdxind2 in gdxind1.groupby(level=plot_fign):
                keys_d[plot_fign] = fig_key
                
                # setup folder
                od = os.path.join(out_dir, fkeys[0], fkeys[1], str(fig_key))
                """
                view(gdxind2)
                gdxind2.index
                fig.show()
                """
                log.info('on %s' % keys_d)
                #===============================================================
                # figure setup
                #===============================================================
                mdex = gdxind2.index
                plt.close('all')
                fig, ax_lib = self.get_matrix_fig(
                                            mdex.unique(plot_rown).tolist(),  # row keys
                                            col_keys,  # col keys
                                            figsize_scaler=4,
                                            constrained_layout=True,
                                            sharey='none',
                                            sharex='none',  # events should all be uniform
                                            fig_id=0,
                                            set_ax_title=True,
                                            )
                
                s = ' '.join(['%s-%s' % (k, v) for k, v in keys_d.items()])
                fig.suptitle('%s Accruacy for %s' % (lossType.upper(), s))
                
                #===============================================================
                # raws
                #===============================================================
                varn = 'grid'
                for ax_key, gser in gdxind2[varn].groupby(level=plot_rown):
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s' % (k, v) for k, v in keys_d.items()])
                    ax = ax_lib[ax_key]['raw']
                    self.ax_hist(ax,
                        gser,
                        label=varn,
                        stat_keys=['min', 'max', 'median', 'mean', 'std'],
                        style_d=dict(color=newColor_d[ax_key]),
                        binWidth=binWidth,
                        plot_zeros=plot_zeros,
                        logger=log.getChild(s1),
                        )
                    
                    # set limits
                    set_lims('raw', ax)
                    
                #===============================================================
                # deltas
                #===============================================================
                varn = 'delta'
                for ax_key, gser in gdxind2[varn].groupby(level=plot_rown):
                    if ax_key == 0:continue
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s' % (k, v) for k, v in keys_d.items()])
                    
                    self.ax_hist(ax_lib[ax_key][varn],
                        gser,
                        label=varn,
                        stat_keys=['min', 'max', 'median', 'mean', 'std'],
                        style_d=dict(color=newColor_d[ax_key]),
                        binWidth=binWidth,
                        plot_zeros=plot_zeros,
                        logger=log.getChild(s1),
                        )
                    
                    upd_lims(varn, ax)
                #===============================================================
                # scatter
                #===============================================================
                for ax_key, gdxind3 in gdxind2.loc[:, ['grid', 'true']].groupby(level=plot_rown):
                    if ax_key == 0:continue
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s' % (k, v) for k, v in keys_d.items()])
                    
                    self.ax_corr_scat(ax_lib[ax_key]['correlation'],
                          
                          gdxind3['grid'].values,  # x (first row is plotting gridded also)
                          gdxind3['true'].values,  # y 
                          style_d=dict(color=newColor_d[ax_key]),
                          label='grid vs true',
                          
                          )
                
                #=======================================================================
                # post formatting
                #=======================================================================
                """
                fig.show()
                """
                for row_key, d0 in ax_lib.items():
                    for col_key, ax in d0.items():
                        
                        # first row
                        if row_key == mdex.unique(plot_rown)[0]:
                            pass
                            
                            # last col
                            if col_key == col_keys[-1]:
                                pass
                            
                        # last row
                        if row_key == mdex.unique(plot_rown)[-1]:
                            # first col
                            if col_key == col_keys[0]:
                                ax.set_xlabel('%s (%s)' % (lossType, 'grid'))
                            elif col_key == col_keys[1]:
                                ax.set_xlabel('%s (%s)' % (lossType, 'delta'))
                            elif col_key == col_keys[-1]:
                                ax.set_xlabel('%s (%s)' % (lossType, 'grid'))
                                 
                        # first col
                        if col_key == col_keys[0]:
                            ax.set_ylabel('count')
                            ax.set_xlim(lim_max_d['raw']['x'])
                            ax.set_ylim(lim_max_d['raw']['y'])
                            
                        # second col
                        if col_key == col_keys[1]:
                            ax.set_ylim(lim_max_d['raw']['y'])
                            
                        # loast col
                        if col_key == col_keys[-1]:
                            # set limits from first columns
                            col1_xlims = ax_lib[row_key]['raw'].get_xlim()
                            ax.set_xlim(col1_xlims)
                            ax.set_ylim(col1_xlims)
                            
                            if not row_key == 0:
                                
                                ax.set_ylabel('%s (%s)' % (lossType, 'true'))
                                # move to the right
                                ax.yaxis.set_label_position("right")
                                ax.yaxis.tick_right()
 
                #===============================================================
                # wrap fig
                #===============================================================
                s = '_'.join([str(e) for e in keys_d.values()])
                self.output_fig(fig, out_dir=od,
                                fname='accuracy_%s_%s_%s' % (lossType, s, self.longname.replace('_', '')),
                                fmt=fmt, logger=log, transparent=False)
                
            #===================================================================
            # wrap folder
            #===================================================================
                
        #===================================================================
        # wrap
        #===================================================================
        log.info('finished')
        
        return

    
    #===========================================================================
    # HELPERS---------
    #===========================================================================
     
    def ax_corr_scat(self,  # correlation scatter plots on an axis
                ax,
                xar, yar,
                label=None,
                
                # plot control
                plot_trend=True,
                plot_11=True,
                
                # lienstyles
                scatter_kwargs={  # default styles
                    
                    } ,
 
                logger=None,
                add_label=True,
                ):
        
        #=======================================================================
        # defaultst
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('ax_corr_scat')
 
        # assert isinstance(stat_keys, list), label
        if not xar.shape == yar.shape:
            raise Error('data mismatch on %s'%(label))
        assert isinstance(scatter_kwargs, dict), label
        #assert isinstance(label, str)
        # log.info('on %s'%data.shape)
        
        #=======================================================================
        # setup 
        #=======================================================================
        max_v = max(max(xar), max(yar))
        xlim = (min(xar), max(xar))
        #=======================================================================
        # add the scatter
        #=======================================================================
        #overwrite defaults with passed kwargs
        scatter_kwargs = {**{'markersize':3.0, 'marker':'.', 'fillstyle':'full'},
                          **scatter_kwargs}
        
        """density color?"""
        ax.plot(xar, yar, linestyle='None', label=label, **scatter_kwargs)
 
        #=======================================================================
        # add the 1:1 line
        #=======================================================================
        if plot_11:
            # draw a 1:1 line
            ax.plot([0, max_v * 10], [0, max_v * 10], color='black', linewidth=0.5, label='1:1')
        
        #=======================================================================
        # add the trend line
        #=======================================================================
        if plot_trend:
            slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(xar, yar)
            
            pearson, pval = scipy.stats.pearsonr(xar, yar)
            
            
            x_vals = np.array(xlim)
            y_vals = intercept + slope * x_vals
            
            ax.plot(x_vals, y_vals, color='red', linewidth=0.5, label='r=%.3f'%rvalue)
 
        #=======================================================================
        # get stats
        #=======================================================================
        
        stat_d = {
                'count':len(xar),
                   'LR.slope':round(slope, 3),
                  # 'LR.intercept':round(intercept, 3),
                  # 'LR.pvalue':round(slope,3),
                  #'pearson':round(pearson, 3), #just teh same as rvalue
                  'r value':round(rvalue, 3),
                   # 'max':round(max_v,3),
                   }
            
        # dump into a string
        
        if add_label:
            annot = label + '\n' + get_dict_str(stat_d)
            
            anno_obj = ax.text(0.1, 0.9, annot, transform=ax.transAxes, va='center')
 
        #=======================================================================
        # post format
        #=======================================================================
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.grid()
        
        return stat_d
                    


    
    def get_confusion(self,
                     df_raw,
                     wetdry=True,
                     logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_confusion')
        
        assert len(df_raw.columns)==2
        
        #=======================================================================
        # prep data
        #=======================================================================
        if wetdry:
            assert (df_raw.dtypes == 'float64').all()
            
            df1 = pd.DataFrame('dry', index=df_raw.index, columns=df_raw.columns)
            
            df1[df_raw>0.0] = 'wet'
            
            labels = ['wet', 'dry']
            
        else:
            raise Error('not impelemented')
            df1 = df_raw.copy()
            
 
        #build matrix
        cm_ar = confusion_matrix(df1['true'], df1['pred'], labels=labels)
        
        cm_df = pd.DataFrame(cm_ar, index=labels, columns=labels)
        
        #convert and label
        
        cm_df2 = cm_df.unstack().rename('counts').to_frame()
        
        cm_df2.index.set_names(['true', 'pred'], inplace=True)
        
        cm_df2['codes'] = ['TP', 'FP', 'FN', 'TN']
        
        cm_df2 = cm_df2.set_index('codes', append=True)
        
        return cm_df, cm_df2.swaplevel(i=0, j=2)
        
        

    def prep_ranges(self, #for multi-simulations, compress each entry using the passed stats 
                    qhi, qlo, drop_zeros, gdx_raw, logger=None):
        if logger is None: logger=self.logger
        log=logger.getChild('prep_ranges')
        #=======================================================================
        # check
        #=======================================================================
        
        assert not gdx_raw.isna().all(axis=1).any(), 'got some assets with all nulls'
        gdx0 = gdx_raw

        
        #check
        assert gdx0.notna().all().all()
        
        
        #===================================================================
        # prep data
        #===================================================================
        #handle zeros
        bx = (gdx0 == 0).all(axis=1)
        if drop_zeros:
            gdx1 = gdx0.loc[~bx, :]
        else:
            gdx1 = gdx0
            
        #collect ranges
        data_d = {'mean':gdx1.mean(axis=1).values}
        #add range for multi-dimensional
        if len(gdx1.columns) > 1:
            data_d.update({
                    'hi':gdx1.quantile(q=qhi, axis=1).values, 
                    'low':gdx1.quantile(q=qlo, axis=1).values})
        return data_d,bx
    
    def get_color_d(self,
                    cvals,
                    colorMap=None,
                    ):
                    
        if colorMap is None: colorMap=self.colorMap
        cmap = plt.cm.get_cmap(name=colorMap) 
        return {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
    
    
    def join_meta_indexers(self, #join some meta keys to the output data
                modelID_l = None, #optional sub-set of modelIDs
                meta_indexers = {'aggLevel', 'dscale_meth'}, #metadat fields to promote to index
                dx_raw=None,
                ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        idn = self.idn
        log = self.logger.getChild('prep_dx')
        if dx_raw is None: 
            dx_raw = self.retrieve('outs')
                
        assert isinstance(meta_indexers, set)
        
        if modelID_l is None:
            modelID_l=dx_raw.index.unique(idn).tolist()
            
        #=======================================================================
        # checks
        #=======================================================================
        
        overlap_l = set(dx_raw.index.names).intersection(meta_indexers)
        if len(overlap_l)>0:
            log.warning('%i requested fields already in the index: %s'%(len(overlap_l), overlap_l))
            for e in overlap_l:
                meta_indexers.remove(e)
        
        #slice to these models
        dx = dx_raw.loc[idx[modelID_l, :,:,:],:].dropna(how='all', axis=1).sort_index(axis=0)
        
        if len(meta_indexers) == 0:
            log.warning('no additional field srequested')
            return dx
        
        cat_df = self.retrieve('catalog')
        
        """
        view(dx.loc[idx[31, :,:,:],:].head(100))
        
        """
        null_cnt_ser = dx.isna().sum(axis=1)
        chk_index = dx.index.copy()
        #=======================================================================
        # check
        #=======================================================================
        miss_l = set(meta_indexers).difference(cat_df.columns)
        assert len(miss_l)==0, 'missing %i requested indexers: %s'%(len(miss_l), miss_l)
        
        miss_l = set(dx_raw.index.unique(idn)).difference(cat_df.index)
        assert len(miss_l)==0
        
        miss_l = set(modelID_l).difference(dx_raw.index.unique(idn))
        assert len(miss_l)==0, '%i/%i requested models not foundin data\n    %s'%(len(miss_l), len(modelID_l), miss_l)
        
 
        #=======================================================================
        # join new indexers
        #=======================================================================
        #slice to selection
        
        cdf1 = cat_df.loc[modelID_l,meta_indexers]
        
        #create expanded mindex from lookups
        assert cdf1.index.name == dx.index.names[0]
        dx.index = dx.index.join(pd.MultiIndex.from_frame(cdf1.reset_index()))
        
        
        
        
        #reorder a bit
        dx = dx.reorder_levels(list(dx_raw.index.names) + list(meta_indexers))
        
        assert_index_equal(dx.index.droplevel(list(meta_indexers)), chk_index)
        
        dx = dx.swaplevel(i=4).sort_index()
        #=======================================================================
        # check
        #=======================================================================
        miss_l = set(meta_indexers).difference(dx.index.names)
        assert len(miss_l)==0
        
        assert np.array_equal(null_cnt_ser.values, dx.isna().sum(axis=1).values)
        
        return dx
    

            
            
            
        
