'''
Created on Jan. 16, 2022

@author: cefect

explore errors in impact estimates as a result of aggregation using hyd model depths
    let's use hp.coms, but not Canflood
    using damage function csvs from figurero2018 (which were pulled from a db)
    intermediate results only available at Session level (combine Study Areas)
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np
import qgis.core

#===============================================================================
# import scipy.stats 
# import scipy.integrate
# print('loaded scipy: %s'%scipy.__version__)
#===============================================================================

start = datetime.datetime.now()
print('start at %s' % start)


 
idx = pd.IndexSlice

#===============================================================================
# setup matplotlib
#===============================================================================
 
import matplotlib
matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt

#set teh styles
plt.style.use('default')

#font
matplotlib_font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 10 #set the figure title size
matplotlib.rcParams['figure.titlesize'] = 12
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)

#===============================================================================
# custom imports--------
#===============================================================================
from hp.oop import Basic, Session, Error
from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf

#===============================================================================
# FUNCTIONS-------
#===============================================================================
def get_pars_from_xls(
        fp = r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\agg - calcs.xlsx',
        sheet_name='hyd.smry',
        ):
    
    df_raw = pd.read_excel(fp, sheet_name=sheet_name, index_col=0)
    
    d = df_raw.to_dict(orient='index')
    
    print('finished on %i \n    %s\n'%(len(d), d))
    
    
class Session(Session, Qproj):
    
    
    def __init__(self, 

                 **kwargs):
        
        #configure handles
        data_retrieve_hndls = {
            'rsamps':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.sa_get(meth='get_rsamps', **kwargs),
                },
            
            'finvg':{
                #'compiled':
                'build':lambda **kwargs: self.sa_finvg(**kwargs),
                },
            
            }
        
        super().__init__(work_dir = r'C:\LS\10_OUT\2112_Agg',
                         data_retrieve_hndls=data_retrieve_hndls,
                         **kwargs)
        
        
    def sa_finvg(self, #build the finv groups on each studyArea
                 dkey=None,
                 
                 **kwargs):
        assert dkey=='finvg'
        
        #run the method on all the studyAreas
        res_d = self.sa_get(meth='get_finvsg', write=False, dkey=dkey, **kwargs)
        
        #write each as a vlay
        
        #save the directory file
        
 
        
    def finvg_compiled(self, #load each compiled finvg from a directory file
                       fp='',
                       ):
        pass
        
        
    def sa_get(self, #spatial tasks on each study area
                       proj_lib={},
                       meth='get_rsamps', #method to run
                       dkey=None,
                       write=True,
                       ):
        #=======================================================================
        # defaults
        #=======================================================================
        
        log = self.logger.getChild('run_studyAreas')
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
                res_d[name] = f()
                
        #=======================================================================
        # write
        #=======================================================================
        if write:
            """frames between study areas dont have any relation to each other... keeping as dict"""
            out_fp = os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname))
            with open(out_fp,  'wb') as f:
                pickle.dump(res_d, f, pickle.HIGHEST_PROTOCOL)
            
    
            
            log.info('finished on %i. wrote to \n    %s'%(len(res_d), out_fp))
        
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
        assert self.idfn in [f.name() for f in finv.fields()]
            
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
    
 
        
                   
                   

    


def run(
        tag='r0',
        proj_lib =     {
            'obwb':{
                  'EPSG': 2955, 
                 'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\inventory\\obwb_2sheds_r1_0106_notShed_cent.gpkg', 
                 'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\dem\\obwb_NHC2020_DEM_20210804_5x5_cmp_aoi04.tif', 
                 'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\wsl\\depth_sB_1218',
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                    }, 
            #===================================================================
            # 'LMFRA': {
            #     'EPSG': 3005, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\finv\\LMFRA_tagComb0612_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\dem\\LMFRA_NHC2019_dtm_5x5_aoi08.tif', 
            #     'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\wd\\0116\\'
            #         }, 
            # 'SaintJohn': {
            #     'EPSG': 3979, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\finv\\microsoft_0517_aoi13_0116.gpkg',
            #      'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\dem\\HRDEM_0513_r5_filnd_aoi12b.tif',
            #       'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\wd\\'
            #                 }, 
            # 'Calgary': {
            #     'EPSG': 3776, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\finv\\calgary_IBI2016_binvRes_170729_aoi02_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\dem\\CoC_WR_DEM_170815_5x5_0126.tif', 
            #     'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\wd\\0116\\'
            #             }, 
            # 'dP': {
            #     'EPSG': 2950, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\finv\\microsoft_20210506_aoi03_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\dem\\HRDEM_CMM2_0722_fild_aoi03_0116.tif', 
            #     'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\wd\\',
            #     },
            },
        
        **kwargs):
    
    with Session(tag=tag,
                 bk_lib = {
                     'rsamps':{'proj_lib':proj_lib},
                     'finvg':{'proj_lib':proj_lib},
                     },
                 **kwargs) as ses:
        
        ses.retrieve('finvg')
 
 
def dev():
    return run(
        compiled_fp_d = {
            'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\Session\r0\20220117\working\rsamps_Session_r0_0117.pickle',
            }
        )
        
    
    
    
if __name__ == "__main__": 
    
 
    output=dev()
 
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))