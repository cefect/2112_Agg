'''
Created on Jan. 18, 2022

@author: cefect



'''
import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np
idx = pd.IndexSlice


from hp.oop import Basic, Session, Error
from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf



class Session(Session, Qproj):
    
    
    def __init__(self, 
                 name='hyd',
                 **kwargs):
        
        #configure handles
        data_retrieve_hndls = {
            'rsamps':{
                #'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.rsamps_sa(**kwargs),
                },
            
            'finvg':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.finvg_sa(**kwargs),
                },
            
            }
        
        super().__init__(work_dir = r'C:\LS\10_OUT\2112_Agg',
                         data_retrieve_hndls=data_retrieve_hndls,name=name,
                         **kwargs)
        
        
    def finvg_sa(self, #build the finv groups on each studyArea
                 dkey=None,
                 
                 **kwargs):
        """here we store the following in one pickle 'finvg'
            fgm_fps_d: filepaths to merged grouped finvs {name:fp}
            fgdir_dxind: directory of finv keys from raw to grouped for each studyArea
            """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('finvg_sa')
        assert dkey=='finvg'
        
        #=======================================================================
        # #run the method on all the studyAreas
        #=======================================================================
        res_d = self.sa_get(meth='get_finvsg', write=False, dkey=dkey, **kwargs)
        
        #unzip results
        finv_gkey_df_d, fgm_vlay_d = dict(), dict()
        for k,v in res_d.items():
            finv_gkey_df_d[k], fgm_vlay_d[k] = v
            
        
        
        #=======================================================================
        # write vector layers
        #=======================================================================
 
        #setup directory
        od = os.path.join(self.wrk_dir, 'fgm_vlays')
        if not os.path.exists(od): os.makedirs(od)
        
        log.info('writing %i layers to %s'%(len(fgm_vlay_d), od))
        
        ofp_d = dict()
        for name, fgm_vlay in fgm_vlay_d.items():
            ofp_d[name] = self.vlay_write(fgm_vlay,
                                     os.path.join(od, 'fgm_%s_%s.gpkg'%(self.longname, name)),
                                     logger=log)
            
        log.debug('wrote %i'%len(ofp_d))
        

        
        #=======================================================================
        # write the dxcol and the fp_d
        #=======================================================================
        dxind = pd.concat(finv_gkey_df_d)
 
        
        log.info('writing \'fgdir_dxind\' (%s) and \'fgm_ofp_d\' (%i) as \'%s\''%(
            str(dxind.shape), len(ofp_d), dkey))
        
        
        #save the directory file
        finvg = {'fgm_ofp_d':ofp_d,'fgdir_dxind':dxind}
        
        self.ofp_d[dkey] = self.write_pick(finvg, 
                           os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),logger=log)
        
        return finvg
        
        
        
        
 
        
 
        
        
    def rsamps_sa(self, #get raster samples for all finvs
                     dkey=None,
                     proj_lib={},
                     ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey=='rsamps', dkey
        log = self.logger.getChild('rsamps_sa')
        
        #=======================================================================
        # child data
        #=======================================================================
        fgm_ofp_d, fgdir_dxind = self.get_finvg()
        
        log.debug('on %i'%len(fgm_ofp_d))
        
        #=======================================================================
        # sample on each
        #=======================================================================
        self.sa_get(proj_lib=proj_lib, meth='get_rsamps', logger=log)
        
        
        return {}
    
    def get_finvg(self):
        
        #=======================================================================
        # load
        #=======================================================================
        d = self.retrieve('finvg')
        
 
        fgm_ofp_d, fgdir_dxind = d.pop('fgm_ofp_d'), d.pop('fgdir_dxind')
        
        #=======================================================================
        # check
        #=======================================================================
        miss_l = set(fgm_ofp_d.keys()).symmetric_difference(fgdir_dxind.index.get_level_values(0).unique())
        assert len(miss_l)==0
        
        for k,fp in fgm_ofp_d.items():
            assert os.path.exists(fp), k
            assert fp.endswith('.gpkg'), k
            
        return fgm_ofp_d, fgdir_dxind
        
        
    def sa_get(self, #spatial tasks on each study area
                       proj_lib={},
                       meth='get_rsamps', #method to run
                       dkey=None,
                       write=True,
                       logger=None,
                       **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('run_studyAreas')
        log.info('on %i \n    %s'%(len(proj_lib), list(proj_lib.keys())))
        
        assert dkey in ['rsamps', 'finvg'], 'bad dkey %s'%dkey
        
        #=======================================================================
        # loop and load
        #=======================================================================
        res_d = dict()
        for i, (name, pars_d) in enumerate(proj_lib.items()):
            log.info('%i/%i on %s'%(i+1, len(proj_lib), name))
            
            with StudyArea(session=self, name=name, tag=self.tag, prec=self.prec,
                           **pars_d) as wrkr:
                
                #raw raster samples
                f = getattr(wrkr, meth)
                res_d[name] = f(**kwargs)
                
        #=======================================================================
        # write
        #=======================================================================
        if write:
            """frames between study areas dont have any relation to each other... keeping as dict"""

            self.write_pick(res_d, os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)), logger=log)
 
        
        return res_d
                

        
class StudyArea(Session, Qproj): #spatial work on study areas
    idfn = 'id' #unique key for assets
    
    def __init__(self, 
                 
                 #pars_lib kwargs
                 EPSG=None,
                 finv_fp=None,
                 dem=None,
                 wd_dir=None,
                 aoi=None,
                 
                 **kwargs):
        
        super().__init__(**kwargs)
        
        #=======================================================================
        # #set crs
        #=======================================================================
        crs = QgsCoordinateReferenceSystem('EPSG:%i'%EPSG)
        assert crs.isValid()
            
        self.qproj.setCrs(crs)
        
 
        #=======================================================================
        # load aoi
        #=======================================================================
        if not aoi is None:
            self.load_aoi(aoi)
            
        #=======================================================================
        # load finv
        #=======================================================================
        finv_raw = self.vlay_load(finv_fp, dropZ=True, reproj=True)
        
        if not self.aoi_vlay is None:
            finv = self.slice_aoi(finv_raw)
            self.mstore.addMapLayer(finv_raw)
        
        else:
            finv = finv_raw
            
 
            
        #check
        assert self.idfn in [f.name() for f in finv.fields()], 'finv \'%s\' missing idfn \"%s\''%(finv.name(), self.idfn)
            
        self.mstore.addMapLayer(finv_raw)
        self.finv_vlay = finv
            
        
            
            
            
            
            
        #=======================================================================
        # attachments
        #=======================================================================
        self.wd_dir=wd_dir
 
        
    def get_finvsg(self, #get the set of finvs per aggLevel
                  finv_vlay=None,
                  grid_sizes = [5, 20, 100], #resolution in meters
                  idfn=None,
                  gcn = 'gid'
                  ):
        """
        
        how do we store an intermitten here?
            study area generates a single layer on local EPSG
            
            Session writes these to a directory
        
 
            
        need a meta table
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_finvs')
        
        if finv_vlay is None: finv_vlay=self.finv_vlay
        if idfn is None: idfn=self.idfn
        
        #=======================================================================
        # loop and aggregate
        #=======================================================================
        fpts_d = dict() #container for resulting finv points layers
        groups_d = dict()
        meta_d = dict()
        log.info('on %i: %s'%(len(grid_sizes), grid_sizes))
        
        for i, grid_size in enumerate(grid_sizes):
            log.info('%i/%i w/ %.1f'%(i+1, len(grid_sizes), grid_size))
            
            #===================================================================
            # #build the grid
            #===================================================================
            gvlay1 = self.creategrid(finv_vlay, spacing=grid_size, logger=log)
            self.mstore.addMapLayer(gvlay1)
            
            #index the grid
            gvlay2 = self.addautoincrement(gvlay1, fieldName='gid', logger=log)
            self.mstore.addMapLayer(gvlay2)
            
            #select those w/ some assets
            gvlay2.removeSelection()
            self.selectbylocation(gvlay2, finv_vlay, logger=log)
            
            #drop these to centroids
            gvlay3 = self.centroids(gvlay2, selected_only=True, logger=log)
            self.mstore.addMapLayer(gvlay3)
            
            #add groupsize field
            
            fpts_d[grid_size] = self.fieldcalculator(gvlay3, grid_size, fieldName='grid_size', fieldType='Integer', logger=log)
            
            #===================================================================
            # #copy/join over the keys
            #===================================================================
            jd = self.joinattributesbylocation(finv_vlay, gvlay2, jvlay_fnl=gcn, method=1, logger=log)
            
            assert jd['JOINED_COUNT']==finv_vlay.dataProvider().featureCount(), 'failed to join some assets'
            jvlay = jd['OUTPUT']
            self.mstore.addMapLayer(jvlay)
            
            df = vlay_get_fdf(jvlay, logger=log).drop('fid', axis=1)
            
            mcnt_ser = df[gcn].groupby(df[gcn]).count()
 
            
            #===================================================================
            # #meta
            #===================================================================
            meta_d[grid_size] = {
                'total_cells':gvlay1.dataProvider().featureCount(), 
                'active_cells':gvlay2.selectedFeatureCount(),
                'max_member_cnt':mcnt_ser.max()
                }
            
            groups_d[grid_size] = df.set_index(idfn)
            
        #=======================================================================
        # combine results----
        #=======================================================================
        log.info('finished on %i'%len(groups_d))
        """
        view(fgm_vlay2)
        """
        #assemble the grid id per raw asset
        finv_gkey_df = pd.concat(groups_d, axis=1).droplevel(axis=1, level=1)
        
        #=======================================================================
        # #merge vector layers
        #=======================================================================
        #add the raw finv (to the head)
        fpts_d = {**{0:self.fieldcalculator(finv_vlay, 0, fieldName='grid_size', fieldType='Integer', logger=log)},
                  **fpts_d}
        
        #merge
        fgm_vlay1 = self.mergevectorlayers(list(fpts_d.values()), logger=log)
        self.mstore.addMapLayer(fgm_vlay1)
        
        #drop some fields
        fnl = [f.name() for f in fgm_vlay1.fields()]
        fgm_vlay2 = self.deletecolumn(fgm_vlay1, list(set(fnl).difference([idfn, gcn, 'grid_size'])), logger=log)
        
        log.info('merged %i gridded inventories into %i pts'%(
            len(fpts_d), fgm_vlay2.dataProvider().featureCount()))
        
        return finv_gkey_df, fgm_vlay2
        
 
            
                
        
    def get_rsamps(self,
                   wd_dir = None,
                   finv_vlay=None,
                   idfn=None,
                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_rsamps')
        if wd_dir is None: wd_dir = self.wd_dir
        if finv_vlay is None: finv_vlay=self.finv_vlay
        if idfn is None: idfn=self.idfn
        
        assert os.path.exists(wd_dir)
        
        #=======================================================================
        # retrieve
        #=======================================================================
        fns = [e for e in os.listdir(wd_dir) if e.endswith('.tif')]
        
        log.info('on %i rlays \n    %s'%(len(fns), fns))
        
        res_d = dict()
        for i, fn in enumerate(fns):
            rname = fn.replace('.tif', '')
            log.debug('%i %s'%(i, fn))
            vlay_samps = self.rastersampling(finv_vlay, os.path.join(wd_dir, fn), logger=log, pfx='samp_')
            self.mstore.addMapLayer(vlay_samps)
            
            ser = vlay_get_fdf(vlay_samps, logger=log).drop('fid', axis=1
                                    ).set_index(idfn).iloc[:,0].rename(rname)
            
            res_d[rname] = ser
            
            """
            view(vlay_samps)
            """
            
        #=======================================================================
        # wrap
        #=======================================================================
        res_df = pd.concat(res_d, axis=1)
        return res_df
    
 
        