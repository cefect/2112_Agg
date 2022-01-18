'''
Created on Jan. 18, 2022

@author: cefect



'''
import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np
idx = pd.IndexSlice



from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error

from hp.basic import set_info

import matplotlib.pyplot as plt

from agg.scripts import Session as agSession


class Session(agSession):
    
    gcn = 'gid'
    
    mindex_dtypes={
                 'studyArea':np.dtype('object'),
                 'id':np.dtype('int64'),
                 'gid':np.dtype('int64'), #both ids are valid
                 'grid_size':np.dtype('int64'), 
                 'event':np.dtype('O'),           
                         }
    
    def __init__(self, 
                 name='hyd',
                 proj_lib={},
                 trim=True, #whether to apply aois
                 **kwargs):
        
    #===========================================================================
    # HANDLES-----------
    #===========================================================================
        #configure handles
        data_retrieve_hndls = {
            'rsamps':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.rsamps_sa(**kwargs),
                },
            
            'finvg':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.finvg_sa(**kwargs),
                },
            'rloss':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.rloss_build(**kwargs),
                },
            'tloss':{
                'build':lambda **kwargs:self.tloss_build(**kwargs),
                },
            
            }
        
        super().__init__( 
                         data_retrieve_hndls=data_retrieve_hndls,name=name,
                         **kwargs)
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.proj_lib=proj_lib
        self.trim=trim
        
    
    #===========================================================================
    # ANALYSIS-------------
    #===========================================================================
    
    def plot_depths(self,
                    xlims = (0,2),
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_depths')
        
        #retrieve child data
        #fgm_ofp_d, fgdir_dxind = self.get_finvg()
        rsdf_d = self.retrieve('rsamps')
        
        log.info('on %i'%len(rsdf_d))
        plt.close('all')
        #=======================================================================
        # loop on studyAreas
        #=======================================================================
         
        for i, (sName, rsamp_dxind) in enumerate(rsdf_d.items()):

            grid_sizes = rsamp_dxind.index.get_level_values(0).unique().tolist()
            
            
            fig, ax_d = self.get_matrix_fig(grid_sizes,list(rsamp_dxind.columns), 
                                        figsize=( len(rsamp_dxind.columns)*3, len(grid_sizes)*3),
                                        constrained_layout=True,
                                        sharey='all', sharex='all', #everything should b euniform
                                        fig_id=i)
            
 
            for grid_size, gdf in rsamp_dxind.groupby(level=0, axis=0): #axis grid rows
                 
                firstCol=True
                for rlayName, ser in gdf.droplevel(0, axis=0).items(): #axis grid columns
                    ax = ax_d[grid_size][rlayName]
                    
                    #plot
                    ar = ser.dropna().values
                    ax.hist(ar, color='blue', alpha=0.3, label=rlayName, density=True, bins=20, range=xlims)
                    
                    #label
                    meta_d = {'grid_size':grid_size,'wet':len(ar), 'dry':ser.isna().sum(), 'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
                    txt = '\n'.join(['%s=%.2f'%(k,v) for k,v in meta_d.items()])
                    ax.text(0.5, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='blue')
                    
                    """
                    plt.show()
                    """
                    
                    #style                    
                    ax.set_xlim(xlims)
                    
                    #first columns
                    if rlayName == gdf.columns[0]:
                        ax.set_ylabel('grid_size=%i'%grid_size)
 
                        
 
                        
                    
                    #first row
                    if grid_size == grid_sizes[0]:
                        ax.set_title('event \'%s\''%(rlayName))
                        
                    #last row
                    if grid_size == grid_sizes[-1]:
                        ax.set_xlabel('depth (m)')
                        
            #===================================================================
            # wrap figure
            #===================================================================
            fig.suptitle('depths for studyArea \'%s\''%sName)
            
            self.output_fig(fig, fname='depths_%s_%s'%(sName, self.longname))
                    
            log.info('finished')
            
            
            """
            plt.show()
            """
            
            
    def plot_tloss(self, #barchart of total losses
                   ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_tloss')
        
        #retrieve child data
        #
        rl_dxind = self.retrieve('tloss')
        
        log.info('on %i'%len(rl_dxind))
        plt.close('all')
        
        #=======================================================================
        # calc total loss
        #=======================================================================
 
        rlT_dxind = rl_dxind.swaplevel(i=0,j=1,axis=0).groupby(level=[0,1,2],axis=0).sum()
    
    #===========================================================================
    # DATA CONSTRUCTION-------------
    #===========================================================================
    def tloss_build(self, #get the total loss
                    dkey=None,
                    **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('tloss_build')
        assert dkey=='tloss'
        gcn = self.gcn
        scale_cn = 'id_cnt'
        #=======================================================================
        # retriever
        #=======================================================================
        rl_dxind = self.retrieve('rloss')
        
        rlnames_d= {lvlName:i for i, lvlName in enumerate(rl_dxind.index.names)}
        
        _, fgdir_dxind = self.get_finvg()
        
        """
        view(rl_dxind)
        view(fgdir_dxind)
        view(ser1)
        """
        #=======================================================================
        # calculate the asset scalers
        #=======================================================================
        grid_d =dict()
        for grid_size, gdf in rl_dxind.groupby(level=rlnames_d['grid_size']):
            #get teh scaler series
            if grid_size==0:
                cnt_serix = pd.Series(1, index=gdf.index).droplevel(level=[rlnames_d['grid_size'], rlnames_d['event']],axis=0)
            else:
                #retrieve the lookup fro these
                ser1 = fgdir_dxind.loc[:, grid_size].rename(gcn)
                assert ser1.notna().all(), grid_size
                
                #get counts
                d = dict()
                for sName, ser2 in ser1.groupby(level=0):                
                    d[sName] = ser2.groupby(ser2).count()
                
                #per-area (for this group size)
                cnt_serix = pd.concat(d)
                
            #wrap
            assert cnt_serix.notna().all(), grid_size
            grid_d[grid_size] = cnt_serix.rename(scale_cn)

        #=======================================================================
        # join
        #=======================================================================
        #shape into a dxind with matching names
        jdxind1 = pd.concat(grid_d, names= rl_dxind.droplevel(rlnames_d['event']).index.names)
        self.check_mindex(jdxind1.index)
        
        jnames_d = {lvlName:i for i, lvlName in enumerate(jdxind1.index.names)}
        assert jdxind1.notna().all()
        
        #check index
        """we expect more in the scaleing lookup (jdxind1) as these have not been filtered for zero impacts"""
        for name in jnames_d.keys():
            
            left_vals = rl_dxind.index.get_level_values(rlnames_d[name]).unique()
            right_vals = jdxind1.index.get_level_values(jnames_d[name]).unique()
            
            el_d = set_info(left_vals, right_vals)
            
            
            
            for k,v in el_d.items():
                log.debug('%s    %s'%(k,v))
                
            assert len(el_d['diff_left'])==0, '%s missing some lookup values'%name
        
        #join to the rloss data
        dxind1 = rl_dxind.join(jdxind1, on=jdxind1.index.names)
        
        assert dxind1[scale_cn].notna().all()
        """
        view(dxind1)
        """
        
        #re-org columns
        rl_cols = rl_dxind.drop('depth',axis=1).columns.tolist()
        
        new_rl_cols= ['%i_rl'%e for e in rl_cols]
        dxind2 = dxind1.rename(columns=dict(zip(rl_cols, new_rl_cols)))
        
        #=======================================================================
        # calc total loss
        #=======================================================================
        tval_colns = ['%i_tl'%e for e in rl_cols]
        rdx= dxind2.loc[:, new_rl_cols].multiply(dxind2[scale_cn], axis=0).rename(
            columns=dict(zip(new_rl_cols, tval_colns)))
        
        dxind3 = dxind2.join(rdx, on=rdx.index.names)
        
        raise Error('stopped here... check that gid.grid_size.studyArea is unique')
 
        
        """
        view(res)
        view(jdxind1)
        view(dxind2)
        """
        
        
        
        
        raise Error('dome')

    
    def rloss_build(self,
                    dkey=None,
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('rloss_build')
        assert dkey=='rloss'
        
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        #depths
        #fgm_ofp_d, fgdir_dxind = self.get_finvg()
        dxser = self.retrieve('rsamps')
        
        #collapse data
        """should probably make this the rsamp format by default"""

        
        log.debug('loaded %i rsamps'%len(dxser))
        
        #vfuncs
        vf_d = self.retrieve('vf_d')
        
        #=======================================================================
        # loop and calc
        #=======================================================================
        log.info('getting impacts from %i vfuncs and %i depths'%(
            len(vf_d), len(dxser)))
            
        res_d = dict()
        for i, (vid, vfunc) in enumerate(vf_d.items()):
            log.info('%i/%i on %s'%(i+1, len(vf_d), vfunc.name))
            
            res_d[vid] = vfunc.get_rloss(dxser.values)
        
        
        #=======================================================================
        # combine
        #=======================================================================
        rdf = pd.DataFrame.from_dict(res_d)
        
        rdf.index = dxser.index
        
        res_dxind = dxser.to_frame().join(rdf)
        
        log.info('finished on %s'%str(rdf.shape))
        
        self.check_mindex(res_dxind.index)
        #=======================================================================
        # write
        #=======================================================================
        self.ofp_d[dkey] = self.write_pick(res_dxind,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                                   logger=log)
        
        return res_dxind
 
                    
    
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
        
        #check index
        dxind.index.set_names('studyArea', level=0, inplace=True)
        self.check_mindex(dxind.index)

 
        
        log.info('writing \'fgdir_dxind\' (%s) and \'fgm_ofp_d\' (%i) as \'%s\''%(
            str(dxind.shape), len(ofp_d), dkey))
        
        
        #save the directory file
        finvg = {'fgm_ofp_d':ofp_d,'fgdir_dxind':dxind}
        
        self.ofp_d[dkey] = self.write_pick(finvg, 
                           os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),logger=log)
        
        return finvg
        
 
    def rsamps_sa(self, #get raster samples for all finvs
                     dkey=None,
                     proj_lib=None,
                     ):
        """
        keeping these as a dict because each studyArea/event is unique
        """
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey=='rsamps', dkey
        log = self.logger.getChild('rsamps_sa')
        if proj_lib is None: proj_lib=self.proj_lib
        gcn=self.gcn
        #=======================================================================
        # child data
        #=======================================================================
        fgm_ofp_d, fgdir_dxind = self.get_finvg()
        
        log.debug('on %i'%len(fgm_ofp_d))
        
        #=======================================================================
        # update the finv
        #=======================================================================
        d = dict()
        for name, pars_d in proj_lib.items():
            d[name] = copy.deepcopy(pars_d)
            d[name]['finv_fp'] = fgm_ofp_d[name]
            d[name]['idfn'] = gcn #these finvs are keyed by gid
        
 
        res_d = self.sa_get(proj_lib=d, meth='get_rsamps', logger=log, dkey=dkey, write=False)
        
        #=======================================================================
        # shape into dxind
        #=======================================================================
        dxind1 = pd.concat(res_d, names=['studyArea', 'grid_size', gcn])
        
 
        
        dxser = dxind1.stack().rename('depth') #promote depths to index
        dxser.index.set_names('event', level=3, inplace=True) 
 
        
        dxser = dxser.swaplevel().swaplevel(i=1,j=0).sort_index(axis=0, level=0, sort_remaining=True)
        
        self.check_mindex(dxser.index)
        
        log.debug('on %i'%len(res_d))
        
        #=======================================================================
        # write
        #=======================================================================
        self.write_pick(dxser, os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                        logger=log)
        
        return dxser
    
    #===========================================================================
    # HELPERS--------
    #===========================================================================
    def get_finvg(self): #get and check the finvg data
        
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
        
    
        
    #===========================================================================
    # GENERICS---------------
    #===========================================================================
    def sa_get(self, #spatial tasks on each study area
                       proj_lib=None,
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
        
        if proj_lib is None: proj_lib=self.proj_lib
        
        log.info('on %i \n    %s'%(len(proj_lib), list(proj_lib.keys())))
        
        
        assert dkey in ['rsamps', 'finvg'], 'bad dkey %s'%dkey
        
        #=======================================================================
        # loop and load
        #=======================================================================
        res_d = dict()
        for i, (name, pars_d) in enumerate(proj_lib.items()):
            log.info('%i/%i on %s'%(i+1, len(proj_lib), name))
            
            with StudyArea(session=self, name=name, tag=self.tag, prec=self.prec,
                           trim=self.trim, out_dir=self.out_dir,
                           **pars_d) as wrkr:
                
                #raw raster samples
                f = getattr(wrkr, meth)
                res_d[name] = f(**kwargs)
                
        #=======================================================================
        # write
        #=======================================================================
        if write:
            """frames between study areas dont have any relation to each other... keeping as dict"""

            self.ofp_d[dkey] = self.write_pick(res_d, os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)), logger=log)
 
        
        return res_d
    
    def check_mindex(self, #check names and types
                     mindex,
                     chk_d = None,
                     logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        #if logger is None: logger=self.logger
        #log=logger.getChild('check_mindex')
        if chk_d is None: chk_d=self.mindex_dtypes
        
        #=======================================================================
        # check
        #=======================================================================
        names_d= {lvlName:i for i, lvlName in enumerate(mindex.names)}
        
        for name, lvl in names_d.items():
 
            assert name in chk_d, 'name \'%s\' not recognized'%name
            assert mindex.get_level_values(lvl).dtype == chk_d[name], \
                'got bad type on \'%s\': %s'%(name, mindex.get_level_values(lvl).dtype.name)
                
        
        return
            
        
 
                

        
class StudyArea(Session, Qproj): #spatial work on study areas
    
    
    def __init__(self, 
                 
                 #pars_lib kwargs
                 EPSG=None,
                 finv_fp=None,
                 dem=None,
                 wd_dir=None,
                 aoi=None,
                 
                 #control
                 trim=True,
                 idfn = 'id', #unique key for assets
                 **kwargs):
        
        super().__init__(**kwargs)
        
        #=======================================================================
        # #set crs
        #=======================================================================
        crs = QgsCoordinateReferenceSystem('EPSG:%i'%EPSG)
        assert crs.isValid()
            
        self.qproj.setCrs(crs)
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.idfn=idfn
        #=======================================================================
        # load aoi
        #=======================================================================
        if not aoi is None and trim:
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
        log = self.logger.getChild('get_finvsg')
        gcn = self.gcn
        if finv_vlay is None: finv_vlay=self.finv_vlay
        if idfn is None: idfn=self.idfn
        
        #=======================================================================
        # loop and aggregate
        #=======================================================================
        fpts_d = dict() #container for resulting finv points layers
        groups_d = dict()
        meta_d = dict()
        log.info('on \'%s\' w/ %i: %s'%(finv_vlay.name(), len(grid_sizes), grid_sizes))
        
        for i, grid_size in enumerate(grid_sizes):
            log.info('%i/%i grid w/ %.1f'%(i+1, len(grid_sizes), grid_size))
            
            #===================================================================
            # #build the grid
            #===================================================================
            
            gvlay1 = self.creategrid(finv_vlay, spacing=grid_size, logger=log)
            self.mstore.addMapLayer(gvlay1)
            log.info('    built grid w/ %i'%gvlay1.dataProvider().featureCount())
            
            #clear all the fields
            #gvlay1a = self.deletecolumn(gvlay1, [f.name() for f in gvlay1.fields()], logger=log)
            
            #index the grid
            #gvlay2 = self.addautoincrement(gvlay1a, fieldName='gid', logger=log)
            gvlay2 = self.renameField(gvlay1, 'id', gcn, logger=log)
            self.mstore.addMapLayer(gvlay2)
            
            #select those w/ some assets
            gvlay2.removeSelection()
            self.createspatialindex(gvlay2)
            log.info('    selecting from grid based on intersect w/ \'%s\''%(finv_vlay.name()))
            self.selectbylocation(gvlay2, finv_vlay, logger=log)
            
            
            gvlay2b = self.saveselectedfeatures(gvlay2, logger=log)
            self.mstore.addMapLayer(gvlay2b)
            
            
            #drop these to centroids
            gvlay3 = self.centroids(gvlay2b, logger=log)
            self.mstore.addMapLayer(gvlay3)
            
            #add groupsize field
            
            fpts_d[grid_size] = self.fieldcalculator(gvlay3, grid_size, fieldName='grid_size', fieldType='Integer', logger=log)
            log.info('    got %i pts from grid'%gvlay3.dataProvider().featureCount())
            #===================================================================
            # #copy/join over the keys
            #===================================================================
            jd = self.joinattributesbylocation(finv_vlay, gvlay2b, jvlay_fnl=gcn, method=1, logger=log)
            
            assert jd['JOINED_COUNT']==finv_vlay.dataProvider().featureCount(), 'failed to join some assets'
            jvlay = jd['OUTPUT']
            self.mstore.addMapLayer(jvlay)
            
            df = vlay_get_fdf(jvlay, logger=log).drop('fid', axis=1)
            
            mcnt_ser = df[gcn].groupby(df[gcn]).count()
            
            """
            view(gvlay1)
            view(fpts_d[grid_size])
            view(gvlay2)
            view(gvlay1a)
            """
 
            
            #===================================================================
            # #meta
            #===================================================================
            meta_d[grid_size] = {
                'total_cells':gvlay1.dataProvider().featureCount(), 
                'active_cells':gvlay2.selectedFeatureCount(),
                'max_member_cnt':mcnt_ser.max()
                }
            
            groups_d[grid_size] = df.set_index(idfn)
            
            log.info('    joined w/ %s'%meta_d[grid_size])
            
        #=======================================================================
        # combine results----
        #=======================================================================
        log.info('finished on %i'%len(groups_d))
        """
        view(finvR_vlay2)
        """
        #assemble the grid id per raw asset
        finv_gkey_df = pd.concat(groups_d, axis=1).droplevel(axis=1, level=1)
        
        #=======================================================================
        # #merge vector layers
        #=======================================================================
        #prep the raw
        finvR_vlay1 = self.fieldcalculator(finv_vlay, 0, fieldName='grid_size', fieldType='Integer', logger=log)
        finvR_vlay2 = self.renameField(finvR_vlay1, idfn, gcn, logger=log) #promote old keys to new keys
        
        #add the raw finv (to the head)
        fpts_d = {**{0:finvR_vlay2},**fpts_d}
        
        #merge
        fgm_vlay1 = self.mergevectorlayers(list(fpts_d.values()), logger=log)
        self.mstore.addMapLayer(fgm_vlay1)
        
        #drop some fields
        fnl = [f.name() for f in fgm_vlay1.fields()]
        fgm_vlay2 = self.deletecolumn(fgm_vlay1, list(set(fnl).difference([gcn, 'grid_size'])), logger=log)
        
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
            
            #retrive and clean
            df = vlay_get_fdf(vlay_samps, logger=log).drop('fid', axis=1).rename(columns={'samp_1':rname})
            
            #force type
            df.loc[:, idfn] = df[idfn].astype(np.int64)
            
            #promote columns to multindex
            df.index = pd.MultiIndex.from_frame(df.loc[:, [idfn, 'grid_size']])
 
            res_d[rname] = df.drop([idfn, 'grid_size'], axis=1)
            
            """
            view(finv_vlay)
            view(vlay_samps)
            """
            
        #=======================================================================
        # wrap
        #=======================================================================
        dxind = pd.concat(res_d, axis=1, keys=None).droplevel(0, axis=1).swaplevel(axis=0)
        
        try:
            self.session.check_mindex(dxind.index)
        except Exception as e:
            raise Error('%s failed index check \n    %s'%(self.name, e))
            
        
        return dxind
    
 
        