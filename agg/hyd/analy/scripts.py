'''
Created on Feb. 21, 2022

@author: cefect
'''
#===============================================================================
# imports------
#===============================================================================
import os, datetime, math, pickle, copy, random, pprint, gc
import matplotlib
import scipy.stats

import pandas as pd
import numpy as np

idx = pd.IndexSlice


from hp.basic import set_info, get_dict_str
from hp.exceptions import Error

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#===============================================================================
# custom imports
#===============================================================================
from hp.plot import Plotr
from hp.oop import Session, Error
from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo, \
    QgsWkbTypes
    
    
from agg.coms.scripts import Catalog



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
 
class ModelAnalysis(Session, Qproj, Plotr): #analysis of model results
    
    colorMap = 'cool'
    
    def __init__(self,
                 catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hyd2\model_run_index.csv',
                 plt=None,
                 name='analy',
                 **kwargs):
        
        data_retrieve_hndls = {
            'outs':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_outs(**kwargs),
                },
            'agg_mindex':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_agg_mindex(**kwargs),
                }
            }
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls,name=name,
                         work_dir = r'C:\LS\10_OUT\2112_Agg',
                         **kwargs)
        self.plt=plt
        self.catalog_fp=catalog_fp
    
    #===========================================================================
    # DATA ANALYSIS---------
    #===========================================================================
    
    def build_outs(self, #collecting outputs from multiple model runs
                    dkey='outs',
 
                     **kwargs):
        assert dkey=='outs'
        
        return self.assemble_model_data(dkey=dkey, **kwargs)
    
    def build_agg_mindex(self,
                         dkey='agg_mindex',
                         **kwargs):
        assert dkey=='agg_mindex'
        
        return self.assemble_model_data(dkey=dkey, **kwargs)
        
    
    
    def assemble_model_data(self, #collecting outputs from multiple model runs
                   modelID_l=[], #set of modelID's to include
                   dkey='outs',
                     catalog_fp=None,
                     load_dkey = 'tloss',
                     write=None,
 
                     ):
        """
        just collecting into a dx for now... now meta
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_%s'%dkey)
        if catalog_fp is None: catalog_fp=self.catalog_fp
        if write is None: write=self.write
        idn = Catalog.idn
        log.info('on %i'%len(modelID_l))
        #=======================================================================
        # retrieve catalog
        #=======================================================================
        cat_df = Catalog(catalog_fp=catalog_fp, logger=log, overwrite=False).get()
        
        #check
        miss_l = set(modelID_l).difference(cat_df.index)
        assert len(miss_l)==0, '%i/%i requested %s not found in catalog:\n    %s'%(
            len(miss_l), len(modelID_l), idn, miss_l)
        
        #=======================================================================
        # load data from modle results
        #=======================================================================
        data_d = dict()
        for modelID, row in cat_df.loc[cat_df.index.isin(modelID_l),:].iterrows():
            log.info('    on %s.%s w/ %i'%(modelID, row['tag'], row['%s_count'%load_dkey]))
            
            #load pickel            
            with open(row['pick_fp'], 'rb') as f:
                data = pickle.load(f)
                
                assert load_dkey in data
                
                data_d[modelID] = data[load_dkey].copy()
                
                del data
                
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
        gc.collect()
        return dx
    
    def build_deltas(self,
                     baseID=0, #modelID to consider 'true'
                     dkey='deltas',
                     dx_raw=None,
                     ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_deltas')
        assert dkey == 'deltas'
        
        if dx_raw is None: dx_raw = self.retrieve('outs')
        dx_agg_mindex = self.retrieve('agg_mindex')
        
        raise Error('stopped here')
 
                
 
        
        
 
        
 
    def build_errs(self,  # get the errors (gridded - true)
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

 
        
 
    def write_loss_smry(self,  # write statistcs on total loss grouped by grid_size, studyArea, and event
                    
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
    
    def write_errs(self,  # write a points layer with the errors
                                       # data control   
                    dkey='errs',
                    
                    # output config
                    # folder1_key = 'studyArea',
                    # folder2_key = 'event',
                    folder_keys=['studyArea', 'event', 'vid'],
                    file_key='grid_size',
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
        log = self.logger.getChild('write_errs')
        assert dkey == 'errs'
        if out_dir is None:out_dir = self.out_dir
            
        gcn = self.gcn
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        # errors
        dx_raw = self.retrieve(dkey)
        # mindex = dx_raw.index
        # names_d= {lvlName:i for i, lvlName in enumerate(dx_raw.index.names)}
        
        # get type
        # dx1 = dx_raw.loc[:, idx[lossType, :,:,:]].droplevel(0, axis=1)
        
        """
        dx_raw.index
        dx_raw.columns
        view(dxind_raw)
        view(dx_raw)
        view(tl_dxind)
        """
        # depths and id_cnt
        tl_dxind = self.retrieve('tloss')
        
        # geometry
        finv_agg_lib = self.retrieve('finv_agg_d')
        
        #=======================================================================
        # prep data
        #=======================================================================
        
        #=======================================================================
        # loop and write
        #=======================================================================
        meta_d = dict()
        # vf_d = self.retrieve('vf_d')
        log.info('on \'%s\' w/ %i' % (dkey, len(dx_raw)))
        
        # lvls = [names_d[k] for k in [folder1_key, folder2_key, file_key]]
        for i, (keys, gdx) in enumerate(dx_raw.groupby(level=folder_keys + [file_key])):
 
            keys_d = dict(zip(folder_keys + [file_key], keys))
            log.debug(keys_d)
                
            #===================================================================
            # retrieve spatial data
            #===================================================================
            # get vlay
            finv_vlay = finv_agg_lib[keys_d['studyArea']][keys_d['grid_size']]
            
            geo_d = vlay_get_geo(finv_vlay)
            fid_gid_d = vlay_get_fdata(finv_vlay, fieldn=gcn)
            #===================================================================
            # prepare data
            #===================================================================
            """layers only support 1d indexers... compressing 2d columsn here"""
            # get column values
            cdf = gdx.columns.to_frame().reset_index(drop=True)            
 
            # get flat frame
            gdf1 = pd.DataFrame(gdx.values,
                index=gdx.index.droplevel(list(keys_d.keys())),
                columns=cdf.iloc[:, 0].str.cat(cdf.iloc[:, 1], sep='.').values,
                ).reset_index()
 
            # reset to fid index
            gdf2 = gdf1.join(pd.Series({v:k for k, v in fid_gid_d.items()}, name='fid'), on='gid').set_index('fid')
            
            """
            view(gdf2)
            """
            #===================================================================
            # build layer
            #===================================================================
            layname = '_'.join([str(e).replace('_', '') for e in keys_d.values()])
            vlay = self.vlay_new_df(gdf2, geo_d=geo_d, layname=layname, logger=log,
                                    crs=finv_vlay.crs(),  # session crs does not match studyAreas
                                    )
            
            #===================================================================
            # write layer
            #===================================================================
            # get output directory
            od = out_dir
            for fkey in [keys_d[k] for k in folder_keys]: 
                od = os.path.join(od, str(fkey))
 
            if not os.path.exists(od):
                os.makedirs(od)
                
            s = '_'.join([str(e) for e in keys_d.values()])
            ofp = os.path.join(od, self.longname + '_' + 'errs_' + s + '.gpkg') 
 
            ofp = self.vlay_write(vlay, ofp, logger=log)
            
            #===================================================================
            # wrap
            #===================================================================
            meta_d[i] = {**keys_d,
                **{ 'len':len(gdf2), 'width':len(gdf2.columns), 'ofp':ofp}
                }
            
        #=======================================================================
        # write meta
        #=======================================================================
        log.info('finished on %i' % len(meta_d))
        
        # write meta
        mdf = pd.DataFrame.from_dict(meta_d, orient='index')
        
        ofp = os.path.join(out_dir, '%s_writeErrs_smry.xls' % self.longname)
        with pd.ExcelWriter(ofp) as writer: 
            mdf.to_excel(writer, sheet_name='smry', index=True, header=True)
            
        return mdf
            
    def get_confusion_matrix(self,  # wet/dry confusion
                             
                             # data control
                             dkey='errs',
                             group_keys=['studyArea', 'event'],
                             pcn='grid_size',  # label for prediction grouping
                             vid=None,  # default is to take first vid
                             
                             # output control
                             out_dir=None,
                             ):
 
        """
        predicteds will always be dry (we've dropped these)
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_confusion_matrix')
        if out_dir is None: out_dir = os.path.join(self.out_dir, 'confusion_mat')
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        labs_d = {1:'wet', 0:'dry'}
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        
        #=======================================================================
        # prep data
        #=======================================================================
        if vid is None: vid = dx_raw.index.unique('vid')[0]  # just taking one function
        dxind1 = dx_raw.loc[idx[:,:,:, vid,:], idx['depth', ['grid', 'true']]
                         ].droplevel(0, axis=1  # clean columns
                                     ).droplevel('vid', axis=0)  # remove other vids
 
        # convert to binary
        bx = dxind1 > 0  # id wets
        
        dxind2 = dxind1.where(~bx, other=labs_d[1]).where(bx, other=labs_d[0])  # replace where false
        
        #=======================================================================
        # loop on each raster
        #=======================================================================
        mat_lib = dict()
        for keys, gdx0 in dxind2.groupby(level=group_keys):
            keys_d = dict(zip(group_keys, keys))
            
            if not keys[0] in mat_lib:
                mat_lib[keys[0]] = dict()
            
            cm_d = dict()
            #===================================================================
            # collect confusion on each grid_size
            #===================================================================
            for pkey, gdx1 in gdx0.groupby(level=pcn, axis=0):
                keys_d[pcn] = pkey
            
                gdf = gdx1.droplevel(level=list(keys_d.keys()), axis=0)
                # if keys_d['grid_size']==0: continue
                cm = confusion_matrix(gdf['true'].values, gdf['grid'].values, labels=list(labs_d.values()))
                
                cm_d[pkey] = pd.DataFrame(cm, index=labs_d.values(), columns=labs_d.values())
                
            #===================================================================
            # combine
            #===================================================================
            dxcol1 = pd.concat(cm_d, axis=1)
            
            # add predicted/true labels
            dx1 = pd.concat([dxcol1], names=['type', 'grid_size', 'exposure'], axis=1, keys=['predicted'])
            dx2 = pd.concat([dx1], keys=['true'])
            
            # store
            mat_lib[keys[0]][keys[1]] = dx2
            
        #===========================================================================
        # write
        #===========================================================================
        for studyArea, df_d in mat_lib.items():
            ofp = os.path.join(out_dir, '%s_confMat_%s.xls' % (studyArea, self.longname))
            
            with pd.ExcelWriter(ofp) as writer:
                for tabnm, df in df_d.items():
                    df.to_excel(writer, sheet_name=tabnm, index=True, header=True)
                    
            log.info('wrote %i sheets to %s' % (len(df_d), ofp))
            
        return mat_lib
    
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
        if colorMap is None: colorMap=self.colorMap
        
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
        
        return self.output_fig(fig, fname='model_smry_%03d_%s' %(modelID, self.longname))
        
 
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
                    plot_coln='event',
                    plot_colr='modelID',
                    plot_bgrp='modelID',
                    
                    #errorbars
                    qhi=0.99, qlo=0.01,
                    
                    #labelling
                    add_label=False,
 
                    
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
        
        if dx_raw is None: dx_raw = self.retrieve('outs')
        if colorMap is None: colorMap=self.colorMap
 
        """
        view(dx)
        dx.loc[idx[0, 'LMFRA', 'LMFRA_0500yr', :], idx['tvals', 0]].sum()
        dx_raw.columns.unique('dkey')
        """
        
        
        log.info('on %s'%str(dx_raw.shape))
        
        #=======================================================================
        # data prep
        #=======================================================================
        #collapse columns
        dx = dx_raw.loc[:, idx[dkey_d.keys(), :]]
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
        # setup the figure
        #=======================================================================
        plt.close('all')
        """
        view(dx)
        plt.show()
        """
 
        fig, ax_d = self.get_matrix_fig(
                                    list(dkey_d.keys()),  
                                    mdex.unique(plot_coln).tolist(),  # col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='row', sharex='all',  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        #fig.suptitle('%s total on %i studyAreas (%s)' % (lossType.upper(), len(mdex.unique('studyArea')), self.tag))
        
        # get colors
        cvals = dx_raw.index.unique(plot_colr)
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
                # data prep
                #===============================================================
                f = getattr(gdx2.groupby(plot_bgrp), dkey_d[keys_d['dkey']])
                
                gdx3 = f().loc[modelID_l, :] #collapse assets, sort
                
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
 
                # labels
                tick_label = [mid_tag_d[mid] for mid in barHeight_ser.index] #label by tag
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
                    color=newColor_d.values(),
                    #label='%s=%s' % (plot_colr, ckey),
                    alpha=0.5,
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
 
                    
                    #===========================================================
                    # #calc errors
                    #===========================================================
                    d = {'pred':tlsum_ser}
                    # get trues
                    d['true'] = gser1r.loc[idx[0, keys_d['studyArea'],:,:, keys_d['vid']]].groupby('event').sum()
                    
                    d['delta'] = (tlsum_ser - d['true']).round(3)
                    
                    # collect
                    tl_df = pd.concat(d, axis=1)
                    
                    tl_df['relErr'] = (tl_df['delta'] / tl_df['true'])
                
                    tl_df['xloc'] = xlocs
                    #===========================================================
                    # add as labels
                    #===========================================================
                    for event, row in tl_df.iterrows():
                        ax.text(row['xloc'], row['pred'] * 1.01, '%+.1f' % (row['relErr'] * 100),
                                ha='center', va='bottom', rotation='vertical',
                                fontsize=10, color='red')
                        
                    log.debug('added error labels \n%s' % tl_df)
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
                    ylabel = '%s (%s)'%(row_key,  dkey_d[keys_d['dkey']])
                    ax.set_ylabel(ylabel)
                    
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        
        return self.output_fig(fig, fname='total_bars_%s' % (self.longname))
                    
 
        
    
    def plot_depths(self,
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

    def plot_tvals(self,
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


    def plot_terrs_box(self,  # boxplot of total errors
                    
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
                ax.grid()
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
 
    def plot_errs_scatter(self,  # scatter plot of error-like data
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
    
    def plot_accuracy_mat(self,  # matrix plot of accuracy
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
                    
    def ax_corr_scat(self,  # correlation scatter plots on an axis
                ax,
                xar, yar,
                label=None,
                
                # plot control
                plot_trend=True,
                plot_11=True,
                
                # lienstyles
                style_d={},
                style2_d={  # default styles
                    'markersize':3.0, 'marker':'.', 'fillstyle':'full'
                    } ,
 
                logger=None,
 
                ):
        
        #=======================================================================
        # defaultst
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('ax_hist')
 
        # assert isinstance(stat_keys, list), label
        assert isinstance(style_d, dict), label
        # log.info('on %s'%data.shape)
 
        #=======================================================================
        # setup 
        #=======================================================================
        max_v = max(max(xar), max(yar))

        xlim = (min(xar), max(xar))
        #=======================================================================
        # add the scatter
        #=======================================================================
        ax.plot(xar, yar, linestyle='None', **style2_d, **style_d)
        """
        view(data)
        self.plt.show()
        """
        
        #=======================================================================
        # add the 1:1 line
        #=======================================================================
        if plot_11:
            # draw a 1:1 line
            ax.plot([0, max_v * 10], [0, max_v * 10], color='black', linewidth=0.5)
        
        #=======================================================================
        # add the trend line
        #=======================================================================
        if plot_trend:
            slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(xar, yar)
            
            pearson, pval = scipy.stats.pearsonr(xar, yar)
            
            x_vals = np.array(xlim)
            y_vals = intercept + slope * x_vals
            
            ax.plot(x_vals, y_vals, color='red', linewidth=0.5)
 
        #=======================================================================
        # get stats
        #=======================================================================
        
        stat_d = {
                'count':len(xar),
                  # 'LR.slope':round(slope, 3),
                  # 'LR.intercept':round(intercept, 3),
                  # 'LR.pvalue':round(slope,3),
                  # 'pearson':round(pearson, 3), #just teh same as rvalue
                  'r value':round(rvalue, 3),
                   # 'max':round(max_v,3),
                   }
            
        # dump into a string
        annot = label + '\n' + '\n'.join(['%s=%s' % (k, v) for k, v in stat_d.items()])
        
        anno_obj = ax.text(0.1, 0.9, annot, transform=ax.transAxes, va='center')
 
        #=======================================================================
        # add grid
        #=======================================================================
        
        ax.grid()
        
        return stat_d
                    
    def ax_hist(self,  # set a histogram on the axis
                ax,
                data_raw,
 
                label='',
                style_d={},
                stat_keys=[],
 
                plot_zeros=False,
                binWidth=None,

                logger=None,
 
                ):
        #=======================================================================
        # defaultst
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('ax_hist')
        assert isinstance(data_raw, pd.Series), label
        assert isinstance(stat_keys, list), label
        assert isinstance(style_d, dict), label
        log.debug('on \'%s\' w/ %s' % (label, str(data_raw.shape)))
        
        #=======================================================================
        # setup  data
        #=======================================================================
        assert data_raw.notna().all().all()
        assert len(data_raw) > 0
 
        dcount = len(data_raw)  # length of raw data
        
        # handle zeros
        bx = data_raw <= 0
        
        if bx.all():
            log.warning('got all zeros!')
            return {}
        
        if plot_zeros: 
            """usually dropping for delta plots"""
            data = data_raw.copy()
        else:
            data = data_raw.loc[~bx]
        
        if data.min() == data.max():
            log.warning('no variance')
            return {}
        #=======================================================================
        # #add the hist
        #=======================================================================
        assert len(data) > 0
        if not binWidth is None:
            try:
                bins = np.arange(data.min(), data.max() + binWidth, binWidth)
            except Exception as e:
                raise Error('faliled to get bin dimensions w/ \n    %s' % e)
        else:
            bins = None
        
        histVals_ar, bins_ar, patches = ax.hist(
            data,
            bins=bins,
            stacked=False, label=label,
            alpha=0.9, **style_d)
 
        # check
        assert len(bins_ar) > 1, '%s only got 1 bin!' % label

        #=======================================================================
        # format ticks
        #=======================================================================
        # ax.set_xticklabels(['%.1f'%value for value in ax.get_xticks()])
            
        #===================================================================
        # #add the summary stats
        #===================================================================
        """
        plt.show()
        """

        bin_width = round(abs(bins_ar[1] - bins_ar[0]), 3)
 
        stat_d = {
            **{'count':dcount,  # real values count
               'zeros (count)':bx.sum(),  # pre-filter 
               'bin width':bin_width,
               # 'bin_max':int(max(histVals_ar)),
               },
            **{k:round(getattr(data_raw, k)(), 3) for k in stat_keys}}
 
        # dump into a string
        annot = label + '\n' + '\n'.join(['%s=%s' % (k, v) for k, v in stat_d.items()])
 
        anno_obj = ax.text(0.5, 0.8, annot, transform=ax.transAxes, va='center')
            
        return stat_d     
