'''
Created on Dec. 31, 2021

@author: cefect

explore impact esitaimtes from damage functions as a result of aggregation
    let's use hp.coms, but not Canflood
    using damage function csvs from figurero2018 (which were pulled from a db)
    
trying a new system for intermediate results data sets
    key each intermediate result with corresponding functions for retriving the data
        build: calculate this data from scratch (and other intermediate data sets)
        compiled: load straight from HD (faster)
        
    in this way, you only need to call the top-level 'build' function for the data set you want
        it should trigger loads on lower results (build or compiled depending on what filepaths have been passed)
'''


#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np
import qgis.core

import scipy.stats 
import scipy.integrate
print('loaded scipy: %s'%scipy.__version__)

start = datetime.datetime.now()
print('start at %s' % start)

from hp.oop import Basic, Error
from hp.pd import view, get_bx_multiVal
 
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

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)


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

class Vfunc(object):
    ycn = 'rl'
    xcn = 'wd'
    
    def __init__(self,
                 vid=0,
                 name='vfunc',
                 logger=None,
                 session=None,
                 meta_d = {}, #optional attributes to attach
                 prec=4, #precsion for general rounding
                 ):
        
        #=======================================================================
        # attachments
        #=======================================================================
        self.vid=vid
        self.name=name
        self.logger = logger.getChild(name)
        self.session=session
        self.meta_d=meta_d
        self.prec=prec
        
        
    def set_ddf(self,
                df_raw,
                logger=None,
                ):
        
        #=======================================================================
        # defai;ts
        #=======================================================================
        if logger is None: logger=self.logger
        #log=logger.getChild('set_ddf')
        
        ycn = self.ycn
        xcn = self.xcn
        
        
        
        
        
        #precheck
        assert isinstance(df_raw, pd.DataFrame)
        l = set(df_raw.columns).symmetric_difference([ycn, xcn])
        assert len(l)==0, l
        
        
        df1 = df_raw.copy().sort_values(xcn).loc[:, [xcn, ycn]]
        
        #check monotoniciy
        for coln, row in df1.items():
            assert np.all(np.diff(row.values)>=0), '%s got non mono-tonic %s vals \n %s'%(
                self.name, coln, df1)
        
        self.ddf = df1
        self.dd_ar = df1.T.values
        #self.ddf_big = df1.copy()
        
        
        #log.info('attached %s'%str(df1.shape))
        
        return self
    
    def get_rloss(self,
                  xar,
                  prec=2,
                  #from_cache=True,
                  ):
        if prec is None: prec=self.prec
        assert isinstance(xar, np.ndarray)
        dd_ar = self.dd_ar
        #clean i tups
        """using a frame to preserve order and resolution"""
        rdf = pd.Series(xar, name='wd_raw').to_frame().sort_values('wd_raw')
        rdf['wd_round'] = rdf['wd_raw'].round(prec)
        

        
 
        
        #=======================================================================
        # identify xvalues to interp
        #=======================================================================
        #get unique values
        xuq = rdf['wd_round'].unique()
        """only interploating for unique values"""
        
        #check with unique values really need to be calculated
        #=======================================================================
        # if from_cache:
        #     bool_ar = np.invert(np.isin(xuq, self.ddf_big[self.xcn].values))
        # else:
        #     bool_ar = np.full(len(xuq), True)
        #=======================================================================
        
        #filter thoe out of bounds
        
        bool_ar = np.full(len(xuq), True)
        xuq1 = xuq[bool_ar]
        #=======================================================================
        # interploate
        #=======================================================================
        dep_ar, dmg_ar = dd_ar[0], dd_ar[1]
        #=======================================================================
        # def get_dmg(x):
        #     return np.interp(x, 
        #=======================================================================
        
        res_ar = np.apply_along_axis(lambda x:np.interp(x,
                                    dep_ar, #depths (xcoords)
                                    dmg_ar, #damages (ycoords)
                                    left=0, #depth below range
                                    right=max(dmg_ar), #depth above range
                                    ),
                            0, xuq1)
        
 
        #=======================================================================
        # plug back in
        #=======================================================================
        """may be slower.. but easier to use pandas for the left join here"""
        rdf = rdf.join(
            pd.Series(res_ar, index=xuq1, name=self.ycn), on='wd_round')
        
        """"
        
        ax =  self.plot()
        
        ax.scatter(rdf['wd_raw'], rdf['rl'], color='black', s=5, marker='x')
        """
        res_ar2 = rdf.sort_index()[self.ycn].values
 
 
        return res_ar2
    
    
    
    
    def plot(self,
             ax=None,
             figNumber=0,
             label=None,
             lineKwargs={},
             logger=None,
             ):
        
        #=======================================================================
        # defautls
        #=======================================================================
        if label is None: label=self.name
        #setup plot
        if ax is None:
            ax = get_ax(figNumber=figNumber)
            
        #get data
        ddf = self.ddf
        xar, yar = ddf.T.values[0], ddf.T.values[1]
        """
        plt.show()
        """
        ax.plot(xar, yar, label=label, **lineKwargs)
        
        return ax
            
                            
 
        
 
        
        
         
 

class Session(Basic):
    vidnm = 'df_id' #indexer for damage functions
    
    data_d = dict()
    
    ycn = 'rl'
    xcn = 'wd'
    colorMap = 'cool'
    
    def __init__(self, 
                  work_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg',
                  mod_name = 'main.py',
                  dfp_d=dict(),
                  
                 **kwargs):
        
        super().__init__(work_dir=work_dir, mod_name=mod_name, 
                         
                         **kwargs)
        
        self.dfp_d=dfp_d
        
        self.data_hndls = { #function mappings for loading data types
                        #compiled takes 1 kwarg (fp)
                        #build takes multi kwargs

                           
            'df_d':{ #no compiled version
                #'compiled':lambda x:self.build_df_d(fp=x),
                'build':lambda **kwargs:self.build_df_d(**kwargs),
                },
            

            
            'vid_df':{
                'build':lambda **kwargs:self.build_vid_df(**kwargs)
                
                },
            'vf_d':{
                'build':lambda **kwargs:self.build_vf_d(**kwargs)
                
                },
            
            'rlMeans_dxcol':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_rlMeans_dxcol(**kwargs),
                },
            'rl_dxcol':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_rl_dxcol(**kwargs),
                },
 
            }
        
        
        

    #===========================================================================
    # COMPMILED-------------
    #===========================================================================

    def get_data(self, #flexible intelligement data retrieval
                 dkey,
                 logger=None,
                 **kwargs
                 ):
        
        if logger is None: logger=self.logger
        log = logger.getChild('get_data')
        
        #get handles
        assert dkey in self.data_hndls, dkey
        
        hndl_d = self.data_hndls[dkey]
        
        #=======================================================================
        # alredy loaded
        #=======================================================================
        if dkey in self.data_d:
            return self.data_d[dkey]
        
        #=======================================================================
        # load by type
        #=======================================================================
        if dkey in self.dfp_d and 'compiled' in hndl_d:
            data = hndl_d['compiled'](fp=self.dfp_d[dkey])
 
        else:
            data = hndl_d['build'](dkey=dkey, **kwargs)
            
        #=======================================================================
        # store
        #=======================================================================
        self.data_d[dkey] = data
            
        log.info('finished on \'%s\' w/ %i'%(dkey, len(data)))
        
        return data
        

    
    def load_aggRes(self,
                    fp):
        
        assert os.path.exists(fp), fp
        assert fp.endswith('.pickle')
        with open(fp, 'rb') as f: 
            dxcol = pickle.load(f)
        
 
        assert isinstance(dxcol, pd.DataFrame), type(dxcol)
        
        return dxcol
    
    #===========================================================================
    # BUILDERS-------------
    #===========================================================================
    def build_df_d(self,
                fp=r'C:\LS\09_REPOS\02_JOBS\2112_Agg\figueiredo2018\cef\csv_dump.xls',
                dkey=None,
                ):
        
        assert dkey=='df_d'
        log = self.logger.getChild('build_df_d')
        df_d = pd.read_excel(fp, sheet_name=None)
        
        log.info('loaded %i pages from \n    %s\n    %s'%(
            len(df_d), fp, list(df_d.keys())))
        
        
        return df_d
        

    

            


        
        
    def build_vid_df(self,
                      df_d = None,
                      vid_l = None,
                     selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[1, 2, 3, 4, 6, 7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     max_mod_cnt = 10, #maximum dfuncs to allow from a single model_id
                     vidnm = None, #indexer for damage functions
                     dkey='vid_df',
                     ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_vid_df')
        assert dkey == 'vid_df'
 
        if vidnm is None: vidnm=self.vidnm
        meta_lib=dict()
        
        
        #=======================================================================
        # loaders
        #=======================================================================
        if df_d is None:
            df_d = self.get_data('df_d')
        
        #=======================================================================
        # identify valid model_ids
        #=======================================================================
        #get metadata for selection
        df1, meta_lib['join_summary'] = self.get_joins(df_d)
        
        log.info('joined %i tabs to get %s' % (len(meta_lib['join_summary']), str(df1.shape)))
        
        if not vid_l is None:
            return df1.loc[vid_l, :]
        
        #id valid df_ids
        bx = get_bx_multiVal(df1, selection_d, log=log)
        
        df2 = df1.loc[bx, :] #.drop(list(selection_d.keys()), axis=1)
        
        log.info('selected %i/%i damageFunctions'%(len(df2), len(df1)))
        
        
        
        #add membership info
        #=======================================================================
        # df3 = df2.copy()
        # for tabName in ['wd', 'fv', 'd', 'relative_loss']:
        #     lkp_df = df_d[tabName].copy()
        #     
        #     df3[tabName] = df3.index.isin(lkp_df[vidnm].unique())
        #     
        # log.debug(df3)
        #=======================================================================
        if not max_mod_cnt is None:
            coln = 'model_id'
            df3 = None
            for k, gdf in df2.groupby(coln):
                if len(gdf)>max_mod_cnt:
                    log.warning('%s=%s got %i models... trimming to %i'%(
                        coln, k, len(gdf), max_mod_cnt))
                    
                    gdf1= gdf.iloc[:max_mod_cnt, :]
                else:
                    gdf1 = gdf.copy()
                    
                if df3 is None:
                    df3 = gdf1
                else:
                    df3 = df3.append(gdf1)
                    
            #wrap
            log.info('cleaned out %i %s'%(len(df2.groupby(coln)), coln))

                     
                    
        else:
            df3 = df2.copy()
        
        """
        view(df2)
        view(df1.loc[bx, :])
        """
        log.info('finished on %s'%str(df3.shape))
        return df3
        

    def build_vf_d(self,
                     vid_df=None,
                     df_d = None,
                     vid_l = None,
 
                     dkey='vf_d',
                     vidnm = None, #indexer for damage functions
                     max_mod_cnt=None,
                     ):
        """
        leaving this as a normal function
            dont want to load class objects from pickles
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_vf_d')
        assert dkey=='vf_d'
        
        if df_d is None: 
            df_d = self.get_data('df_d')
            
        if vid_df is None: 
            vid_df=self.get_data('vid_df', vid_l=vid_l, max_mod_cnt=max_mod_cnt)
            
        if vid_l is None: 
            vid_l=vid_df.index.tolist()
            
        assert len(vid_l)==len(vid_df)
        
            
        if vidnm is None: vidnm=self.vidnm
        xcn = self.xcn
        ycn = self.ycn
            
        
        #=======================================================================
        # spawn each df_id
        #=======================================================================
        def get_wContainer(tabName):
            df1 = df_d[tabName].copy()
            jdf = df_d[tabName+'_container'].copy()
            
            jcoln = jdf.columns[0]
            
            return df1.join(jdf.set_index(jcoln), on=jcoln) 
 
        lkp_d = {k:get_wContainer(k) for k in ['wd', 'relative_loss']}
 
        
        
        def get_by_vid(vid, k):
            df1 = lkp_d[k].groupby(vidnm).get_group(vid).drop(vidnm, axis=1).reset_index(drop=True)
            #drop indexers
            bxcol = df1.columns.str.contains('_id')
            return df1.loc[:, ~bxcol]
        
 
        vf_d = dict()
        log.info('spawning %i dfunvs'%len(vid_df))
        for i, (vid, row) in enumerate(vid_df.iterrows()):
            #log = self.logger.getChild('spawn_%i'%vid)
            #log.info('%i/%i'%(i, len(df2)))
            
            #===================================================================
            # #get dep-damage
            #===================================================================
            
            wdi_df = get_by_vid(vid, 'wd')
            rli_df = get_by_vid(vid, 'relative_loss')
            
            assert np.array_equal(wdi_df.index, rli_df.index)
            
            ddf =  wdi_df.join(rli_df).rename(columns={
                'wd_value':'wd', 'relative_loss_value':'rl'})
            #===================================================================
            # spawn
            #===================================================================
            vf_d[vid] = Vfunc(
                vid=vid,
                name=row['abbreviation']+'_%03i'%vid,
                logger=self.logger, session=self,
                meta_d = row.dropna().to_dict(), 
                ).set_ddf(ddf)
                
            #===================================================================
            # check
            #===================================================================
            for attn in ['xcn', 'ycn']:
                assert getattr(self, attn) == getattr(vf_d[vid], attn), attn
            
                
        log.info('spawned %i vfuncs'%len(vf_d))
        
        #self.vf_d = vf_d
        
        return vf_d
    
    def build_rlMeans_dxcol(self,  #get aggregation erros for a single vfunc
                 vf_d=None,
                 vid_l=[],
                 dkey=None,
                 
                 #vid_df keys
                 max_mod_cnt=None,
                 chunk_size=1,
                 **kwargs
                 ):
        
        log = self.logger.getChild('r')
        assert dkey== 'rlMeans_dxcol'
 
        #=======================================================================
        # #load dfuncs
        #=======================================================================
        if vf_d is None:
            vf_d = self.get_data('vf_d', vid_l=vid_l, max_mod_cnt=max_mod_cnt)
            
            
        #=======================================================================
        # setup
        #=======================================================================
        out_dir = os.path.join(self.out_dir, dkey)
        if not os.path.exists(out_dir): 
            os.makedirs(out_dir)
 
        #======================================================================
        # calc for each
        #======================================================================
        plt.close('all')
        
        dump_fp_d, res_lib, master_vids, err_d = dict(), dict(), dict(), dict()
 
        for i, (vid, vfunc) in enumerate(vf_d.items()):
            
            try:
                res_lib[vfunc.vid] = self.get_rl_mean_agg(vfunc,logger=log.getChild(str(vid)), **kwargs)
                master_vids[vid] = vfunc.vid
            except Exception as e:
                msg = 'failed on %i w/ %s'%(vid, e)
                log.error(msg)
                err_d[i] = msg
            
            
            #write cunks
            if i%chunk_size==0 and i>0: 
                if i +1 == len(vf_d): continue #skip the last
                j = int(i/chunk_size) #chunk count
                dump_fp_d[j] = os.path.join(out_dir, 'dump_%04i.pickle'%i)
                
                with open(dump_fp_d[j], 'wb') as f:
                    pickle.dump(res_lib, f, pickle.HIGHEST_PROTOCOL)
                    
                log.info('i=%i j=%i and %i entries wrote dump '%(i, j,len(res_lib))) 
                    
                del res_lib
                #reset
                res_lib = dict()
            
        """
        i=10
        """
        #=======================================================================
        # re-assemble
        #=======================================================================
        if len(dump_fp_d)>0:
            #load each
            log.info('loading %i dumps'%len(dump_fp_d))
            for k, fp in dump_fp_d.items():
                
                #load the dump
                with open(fp, 'rb') as f: 
                    rdi = pickle.load(f)
                    
                #add back in
                l = set(res_lib.keys()).intersection(rdi.keys())
                assert len(l) == 0, l
                
                res_lib.update(rdi)
                
                """
                rdi.keys()
                res_lib.keys()
                
                """
 
 
        #check
        l = set(master_vids.values()).symmetric_difference(res_lib.keys())
        assert len(l)==0, l
            
                
        #=======================================================================
        # write
        #=======================================================================
        dxcol = pd.concat(res_lib, axis=1, 
                              names=[self.vidnm] + list(res_lib[vfunc.vid].columns.names))
            
        out_fp = self.write_dxcol(dxcol, dkey, logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        for k, msg in err_d.items():
            log.error(msg)
        
 
        raise Error('')
 
        return dxcol
    
 
    def build_rl_dxcol(self, #combine discretized xmean results onto single domain
                        dxcol_raw=None,
                        vf_d = None,
                        vid_l=[],
                        plot_meanCalcs=True, #plot calc loops of ger_rl_mean_agg
                        dkey='rl_dxcol',
                        **kwargs
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild(dkey)
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol_raw is None:
            dxcol_raw = self.get_data('rlMeans_dxcol', vid_l=vid_l, plot=plot_meanCalcs)
            
        
        #=======================================================================
        # check
        #=======================================================================
        #check index
        mdex = dxcol_raw.columns
        assert np.array_equal(np.array(mdex.names), np.array([vidnm, 'xmean', 'xvar', 'aggLevel', 'vars']))
        
            
        #check lvl0:vid
        res_vids_l = mdex.get_level_values(0).unique().tolist()
        l = set(vid_l).difference(res_vids_l)
        assert len(l) == 0, '%i requested vids not in the results: %s'%(len(l), l)
        
        #check lvl4:vars
        l = set(mdex.get_level_values(4).unique()).symmetric_difference([xcn, ycn])
        assert len(l)==0, l
        
        #check lvl3:aggLevel
        assert 'int' in mdex.get_level_values(3).unique().dtype.name
        

        
        #=======================================================================
        # #trim to selection
        #=======================================================================
        dxcol = dxcol_raw.sort_index(axis=1)
        
        dxcol = dxcol.loc[:, idx[vid_l, :,:, :]]
        mdex = dxcol.columns
        
        #check
        l = set(mdex.get_level_values(0).unique().tolist()).symmetric_difference(vid_l)
        assert len(l)==0
        
        
        log.info('loaded  rlMeans_dxcol w/ %s'%str(dxcol.shape))
        #=======================================================================
        # get vfuncs-----
        #=======================================================================
        if vf_d is None: 
            vf_d =  self.get_data('vf_d', vid_l=vid_l)
            
        #check
        l = set(vf_d.keys()).symmetric_difference(vid_l)
        assert len(l)==0, l
 
        #=======================================================================
        # loop on xmean--------
        #=======================================================================
        
        log.info('on %i vfuncs: %s'%(len(vid_l), vid_l))
        res_d = dict()
        for vid, gdf0 in dxcol.groupby(level=0, axis=1):
            
            res_d[vid] = self.calc_rlMeans(gdf0.droplevel(0, axis=1), vf_d[vid], **kwargs)
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        """this is a collapsed version of the rdxcol... wich only mean rloss values"""
        rdxcol = pd.concat(res_d, axis=1, names=[vidnm]+ list(res_d[vid].columns.names))
        self.write_dxcol(rdxcol, dkey,  logger=log)
        
        log.info('finished w/ %s'%str(rdxcol.shape))
        return rdxcol


            
 

 
    #===========================================================================
    # TOP RUNNERS---------
    #===========================================================================
    
    def run_dd_agg(self, #integrate delta areas
                        dxcol_raw=None, #depth-damage at different AggLevel and xvars
 
                        vid_l=[],
 
                        #dkey='rl_dxcol',
                        **kwargs
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('rdd')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol_raw is None:
            dxcol_raw = self.get_data('rl_dxcol', vid_l=vid_l, **kwargs)
            
        
        #=======================================================================
        # check
        #=======================================================================
        #check index
        mdex = dxcol_raw.columns
        assert np.array_equal(np.array(mdex.names), np.array([vidnm, 'xvar', 'aggLevel']))
        
        
        #=======================================================================
        # #trim to selection
        #=======================================================================
        dxcol = dxcol_raw.sort_index(axis=1)

        dxcol = dxcol.loc[:, idx[vid_l, :,:]]
        mdex = dxcol.columns
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}
        
        #check
        l = set(mdex.get_level_values(0).unique().tolist()).symmetric_difference(vid_l)
        assert len(l)==0
        
        
        log.info('loaded  rl_dxcol w/ %s'%str(dxcol.shape))
        """
        view(dxcol)
        """
        #=======================================================================
        # loop on vid--------
        #=======================================================================
        plt.close('all')
        log.info('on %i vfuncs: %s'%(len(vid_l), vid_l))
        res_d = dict()
        for vid, gdf0 in dxcol.groupby(level=0, axis=1):
 
            
            res_d[vid] = self.calc_areas(gdf0.droplevel(0, axis=1), logger=log.getChild('v%i'%vid))
            
        #=======================================================================
        # get stats on set
        #=======================================================================
        log.info('got %i'%len(res_d))
        rdxcol = pd.concat(res_d, axis=1, names=names_d.keys())
        
        #move the vid to the index
        dx1 = rdxcol.T.unstack(level=vidnm).T.swaplevel(axis=0).sort_index(level=0)
        
        #calc each
        d = dict()
        for stat in ['mean', 'min', 'max']:
            d[stat] = getattr(dx1, stat)()
            
        #add back
        dx2 = dx1.append(pd.concat([pd.concat(d, axis=1).T], keys=['stats']))
            
        
        #=======================================================================
        # write
        #=======================================================================
        self.write_dxcol(dx2, 'agg_areas')
        
        return dx2
        """
        view(rdxcol)
        """
        


 
            
 
    def plot_all(self,
                 #data selection
                 vid_l=None,
                 
                 vf_d=None,
                 vid_df=None,
                 
 
                 
                 #formatting
                 phndl_d=None, #optional plot handles
                 xlims = None,ylims=None,
                 figsize=(10,10),
                 lineKwargs = { #global kwargs
                     'linewidth':0.5
                     },
                 
                 
                 
                 title=None,
                 ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_all')
 
        vidnm=self.vidnm
        
        if vf_d is None: 
            vf_d=self.get_data('vf_d', vid_l=vid_l, vid_df=vid_df)
        
        log.info('on %i'%len(vf_d))
        
        if phndl_d is None:
            phndl_d = {k:{} for k in vf_d.keys()}
            
        
        #=======================================================================
        # setup figure
        #=======================================================================
        plt.close('all')
        fig = plt.figure(0,
            figsize=figsize,
            tight_layout=False,
            constrained_layout=True)
        
        ax = fig.add_subplot(111)
        
        """
        plt.show()
        phndl_d.keys()
        """
        
        
        
        for vid, vfunc in vf_d.items():
            log.info('on %s'%vfunc.name)
            
            if not vid in phndl_d:
                log.warning('%s missing handles'%vfunc.name)
                phndl_d[vid]=dict()
 
            
            vfunc.plot(ax=ax, lineKwargs = {**lineKwargs, **phndl_d[vid]}, logger=log)

            
        log.info('added %i lines'%len(vf_d))
        
        #=======================================================================
        # style
        #=======================================================================
        xvar, yvar = vfunc.xcn, vfunc.ycn
        
        if not xlims is None:
            ax.set_xlim(xlims)
        if not ylims is None:
            ax.set_ylim(ylims)
        ax.set_xlabel('\'%s\' water depth (m)'%xvar)
        ax.set_ylabel('\'%s\' relative loss (pct)'%yvar)
        ax.grid()
 
        if title is None:
            title = '\'%s\' vs \'%s\' on %i'%(xvar, yvar, len(vf_d))
        ax.set_title(title)
        
        #legend
        #fig.legend()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        #=======================================================================
        # write figure
        #=======================================================================
        return fig
    
    #===========================================================================
    # HELPERS---------
    #===========================================================================
    
    def calc_areas(self, #calc areas for a single vfunc
            dxcol,
            logger=None,
            ):
        """
        
        not setup super well... we're also calcing this in calc_rlMeans to get the nice bar charts
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('calc_area')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        
        mdex = dxcol.columns
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}
        #=======================================================================
        # loop on xvar
        #=======================================================================
        res_lib = dict()
        for xvar, gdf0 in dxcol.groupby(level=names_d['xvar'], axis=1):
            res_lib[xvar] = dict()
            for aggLevel, gdf1 in gdf0.groupby(level=names_d['aggLevel'], axis=1):
                
                #===============================================================
                # get data
                #===============================================================
                
                #retrieve array
                yar = gdf1.droplevel(level=0, axis=1).iloc[:, 0].values
                

                #get base
                if aggLevel==0:
                    xar = gdf1.droplevel(level=0, axis=1).index.values
                    yar_base = yar
                    continue
                
                #get deltas
                diff_ar = yar_base - yar
            
                res_d = dict()
                #===============================================================
                # total area
                #===============================================================
                """negatives often balance positives"""

                
                #integrate
                res_d['total'] = scipy.integrate.trapezoid(diff_ar, xar)
                
                #===============================================================
                # positives areas
                #===============================================================
                #clear out negatives
                diff1_ar = diff_ar.copy()
                diff1_ar[diff_ar<=0] = 0
                
                res_d['positives'] = scipy.integrate.trapezoid(diff1_ar, xar)
                
                #===============================================================
                # negative  areas
                #===============================================================
                #clear out negatives
                diff2_ar = diff_ar.copy()
                diff2_ar[diff_ar>=0] = 0
                
                res_d['negatives'] = scipy.integrate.trapezoid(diff2_ar, xar)
                
                #===============================================================
                # wrap
                #===============================================================
                res_lib[xvar][aggLevel] = res_d
                
                """
                plt.plot(xar, diff_ar)
                plt.plot(xar, diff1_ar)
                plt.plot(xar, diff2_ar)
                """
            #===================================================================
            # wrap xvar
            #===================================================================
            res_lib[xvar] = pd.DataFrame.from_dict(res_lib[xvar])
                
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i'%len(res_lib))
        
        rdxcol = pd.concat(res_lib, axis=1)
        
        return rdxcol
        
 
            
            
            
    def calc_rlMeans(self, #impact vs depth analysis on multiple agg levels for a single vfunc
                     dxcol, vfunc,
                     
                     quantiles = (0.05, 0.95), #rloss qantiles to plot
                     
                     ylims_d = {
                         'dd':(0,60), 'delta':(-15,15), 'bars':(-15,15)
                         },
                     
                     xlims = (0,2),
 
                     
                     #plot style
                     colorMap=None,
                     prec=4, #rounding table values
                     title=None,
                     
                     ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('calc_rl_v%s'%vfunc.vid)
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        if colorMap is None: colorMap=self.colorMap
 
        plt.close('all')
 
        #=======================================================================
        # check mdex
        #=======================================================================
        mdex = dxcol.columns
        assert np.array_equal(np.array(mdex.names), np.array(['xmean', 'xvar', 'aggLevel', 'vars']))
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}


        
        #=======================================================================
        # #retrieve domain info
        #=======================================================================
        lVals_d = dict()
        for lvlName, lvl in names_d.items():
            lVals_d[lvlName] = np.array(dxcol.columns.get_level_values(lvl).unique())
 
        log.info('for ' + ', '.join(['%i %sz'%(len(v), k) for k,v in lVals_d.items()]))
        #===================================================================
        # setup figure
        #===================================================================
        
        
        #get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap)
        l = lVals_d['aggLevel']        
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(l, np.linspace(0, 1, len(l)))).items()}
        
        

        
        fig, ax_d = self.get_matrix_fig(list(ylims_d.keys()), lVals_d['xvar'].tolist(), 
                                        figsize=(len(lVals_d['xvar'])*4,3*4),
                                        constrained_layout=True)
        
 
        if title is None:
            title = vfunc.name
        fig.suptitle(title)
        """
        fig.show()
        """

        """
        fig.show()
 
        """
        #===================================================================
        # loop by xvar--------
        #===================================================================
        res_lib = dict()
        """splitting this by axis... so needs to be top for loop"""
        for xvar, gdf0 in dxcol.groupby(level=names_d['xvar'], axis=1):
            
            
            """
            view(gdf0.droplevel(names_d['xvar'], axis=1))
            """
            
            #===================================================================
            # loop aggleve
            #===================================================================
            """probably best to loop this next as they'll share plotting pars"""
            res_d, area_lib = dict(), dict()
            for aggLevel, gdf1 in gdf0.groupby(level=names_d['aggLevel'], axis=1):
                
                #===============================================================
                # plotting defaults
                #===============================================================
                color = newColor_d[aggLevel]
                
                #===============================================================
                # setup data
                #===============================================================
                #get a clean frame for this aggLevel
                rdXmean_dxcol = gdf1.droplevel(level=[names_d['aggLevel'], names_d['xvar']], axis=1
                                           ).dropna(how='all', axis=0)
                
                #drop waterDepth info
                rXmean_df = rdXmean_dxcol.loc[:,idx[:,ycn]].droplevel(1, axis=1)
                
                #===============================================================
                # mean impacts per depth 
                #===============================================================
                
                #calc
                rser1 = rXmean_df.mean(axis=0)
                res_d[aggLevel] = rser1.copy()
                
                #plot
                ax = ax_d['dd'][xvar]
                pkwargs = dict( color=color, linewidth=0.5, marker='x', markersize=3, alpha=0.8) #using below also
                ax.plot(rser1,label='aggLevel=%i (rl_mean)'%aggLevel, **pkwargs)
                
                #===============================================================
                # #quartiles
                #===============================================================
                    
                if not quantiles is None:
                    ax.fill_between(
                        rser1.index, #xvalues
                         rXmean_df.quantile(q=quantiles[1], axis=0).values, #top of hatch
                         rXmean_df.quantile(q=quantiles[0], axis=0).values, #bottom of hatch
                         color=color, alpha=0.1, hatch=None)
                    
                #wrap impacts-depth
                ax.legend()
                
                #===============================================================
                # deltas----------
                #===============================================================
                #===============================================================
                # get data
                #===============================================================
                #retrieve array
                yar = rser1.values
                
                #get base
                if aggLevel==0:
                    xar = rser1.index.values
                    yar_base = yar
                    continue
                
                #get deltas
                diff_ar = yar_base - yar
                
                #===============================================================
                # plot
                #===============================================================
                ax = ax_d['delta'][xvar]
                ax.plot(xar, diff_ar, label='aggLevel=%i (true - rl_mean)'%aggLevel, **pkwargs)
                #===============================================================
                # areas----
                #===============================================================
                area_d = dict()
                """negatives often balance positives
                fig.show()
                """

                
                #integrate
                area_d['total'] = scipy.integrate.trapezoid(diff_ar, xar)
                
                #===============================================================
                # positives areas
                #===============================================================
                #clear out negatives
                diff1_ar = diff_ar.copy()
                diff1_ar[diff_ar<=0] = 0
                
                area_d['positive'] = scipy.integrate.trapezoid(diff1_ar, xar)
                
                #===============================================================
                # negative  areas
                #===============================================================
                #clear out negatives
                diff2_ar = diff_ar.copy()
                diff2_ar[diff_ar>=0] = 0
                
                area_d['negative'] = scipy.integrate.trapezoid(diff2_ar, xar)
                
                #ax.legend()
                
                #===============================================================
                # wrap agg level
                #===============================================================
                area_lib[aggLevel] = area_d
                
                log.debug('finished aggLevel=%i'%aggLevel)
            
            ax.legend() #turn on the diff legend
            #===================================================================
            # add area bar charts
            #===================================================================
            ax = ax_d['bars'][xvar]
            
            adf = pd.DataFrame.from_dict(area_lib).round(prec)
            adf.columns = ['aL=%i'%k for k in adf.columns]
            assert adf.notna().all().all()
            
            locs = np.linspace(.1,.9,len(adf))
            for i, (label, row) in enumerate(adf.iterrows()):
                ax.bar(locs+i*0.1, row.values, label=label, width=0.1, alpha=0.5,
                       color={'total':'black', 'positive':'orange', 'negative':'blue'}[label],
                       tick_label=row.index)
                
            
            
            #first row
            if xvar == min(lVals_d['xvar']):
                ax.set_ylabel('area under rl difference curve')
                ax.legend()
            
            #ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
            
            """
            fig.show()
            ax.clear()
            """
            #===================================================================
            # #set a tittle above the table
            # ax.set_title('difference areas', y=0.8)
            # ax.set_axis_off()
            # 
            # ax.table(rowLabels=adf.index, colLabels=adf.columns, cellText=adf.values.tolist(), loc='center', colLoc='center')
            #===================================================================
            #===================================================================
            # #wrap xvar
            #===================================================================
            res_lib[xvar] = pd.concat(res_d, axis=1)
            log.info('finished xvar=%.2f'%xvar)
            
        #=======================================================================
        # post plot --------
        #=======================================================================
        #===================================================================
        # draw the vfunc  and setup the axis
        #===================================================================
 
            
        lineKwargs = {'color':'black', 'linewidth':1.0, 'linestyle':'dashed'}
 
        #=======================================================================
        # #depth-damage
        #=======================================================================
        first=True
        for xvar, ax in ax_d['dd'].items():
        
            vfunc.plot(ax=ax, label=vfunc.name, lineKwargs=lineKwargs)
            ax.set_title('xvar = %.2f'%xvar)
            ax.set_xlabel(xcn)
            ax.grid()
            
            if first:
                ax.set_ylabel(ycn)
                first=False
            

        #=======================================================================
        # #delta
        #=======================================================================
        first=True
        for xvar, ax in ax_d['delta'].items():
            ax.set_xlabel(xcn)
            ax.grid()
            
            ax.hlines(0, -5, 5, label='true', **lineKwargs)
 
            
            if first:
                ax.set_ylabel('%s (true - mean)'%ycn)
                first=False
 
                        
            #ax.legend() #need to turn on at the end
            
        #set limits
        for rowName in ax_d.keys():
            for colName, ax in ax_d[rowName].items():
                
                #xlims
                if not rowName == 'bars' and not xlims is None:
                    ax.set_xlim(xlims)
                    
                #ylims
                if not ylims_d[rowName] is None:
                    ax.set_ylim( ylims_d[rowName])
                
 
                
                
         
        #=======================================================================
        # write data
        #=======================================================================
        newNames_l = [{v:k for k,v in names_d.items()}[lvl] for lvl in [1,2]]  #retrieve labes from original dxcol
        
        
        res_dxcol = pd.concat(res_lib, axis=1, names=newNames_l)
        
        self.output_fig(fig, fname='%s_%s_rlMeans'%(self.resname, vfunc.vid))
        
        return res_dxcol
                
 

    def get_rl_mean_agg(self, #calculate rloss at different levels of aggregation
                          vfunc,
                          
                          #aggrevation levels to consider
                          aggLevels_l= [2, 
                                        5, 100,
                                        ],
                          
                          
                          #xvalues to iterate (xmean)
                          xdomain=(0,2), #min/max of the xdomain
                          xdomain_res = 30, #number of increments for the xdomain
                          

                          
                          #random depths pramaeters
                          xvars_ar = np.linspace(.1,1,num=4), #varice values to iterate over
                          statsFunc = scipy.stats.norm, #function to use for the depths distribution
                          depths_resolution=2000,  #number of depths to draw from the depths distritupion
                          
                          #plotting parameters
                        colorMap = None,
                        plot=False,
                        logger=None,
                          ):

        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_rl_mean')
        log.info('on %s'%vfunc.name)
        
        if colorMap is None: colorMap=self.colorMap
        xcn, ycn = self.xcn, self.ycn
        
        
        
        fig_lib, meta_lib, res_lib, ax_lib = dict(), dict(), dict(), dict()
        
        #get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap)        
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(aggLevels_l, np.linspace(0, 1, len(aggLevels_l)))).items()}
        
 
        #xaxis control
        self.master_xlim = (0,0) #set for controlling bounds
        
        def upd_xlim(ax):
            new = ax.get_xlim()
            
            old = copy.copy(self.master_xlim)
            self.master_xlim = (
                                round(min(old[0], new[0]), 2),
                               round(max(old[1], new[1]), 2)
                               )
            
        
        #=======================================================================
        # loop mean deths 
        #=======================================================================
        xmean_ar = np.linspace(xdomain[0],xdomain[1],num=xdomain_res)
        for i, xmean in enumerate(xmean_ar):
            log.info('%i/%i xmean=%.2f\n'%(i, len(xmean_ar), xmean))
            
            #===================================================================
            # setup working fig
            #===================================================================
            
            fig, ax_d = self.get_matrix_fig(['depths_dist', 'rl_scatter','rl_box'], xvars_ar.tolist(), 
                                            figsize=(4*len(xvars_ar),8),fig_id=i)
        
            ax_lib[xmean] = ax_d
            
            """
            plt.show()
            """
            
            #===================================================================
            # #depth variances (for generating depth values)-------
            #===================================================================
            meta_lib[xmean], res_lib[xmean] = dict(), dict() #add pages
            for xvar in xvars_ar:
                res_d = dict()
                title = 'xmean=%.2f xvar=%.2f'%(xmean, xvar)
                log.info(title)
                
                #===============================================================
                # get depths
                #===============================================================
                #build depths distribution
                ax = ax_d['depths_dist'][xvar]
                rv = statsFunc(loc = xmean, scale = xvar)
                self.plot_rv(rv, ax=ax, xlab='depth (m)', title=title)
                
                #get depths set
                depths_ar = rv.rvs(size=depths_resolution)
                ax.hist(depths_ar, color='blue', alpha=0.5, label='sample %i'%depths_resolution, density=True, bins=20)
                ax.text(0.5, 0.9, '%i samples'%depths_resolution, transform=ax.transAxes, va='center',fontsize=8, color='blue')
                upd_xlim(ax)
 
                #===============================================================
                # #get average impact (w/o aggregation)
                #===============================================================
                ax = ax_d['rl_scatter'][xvar]
                res_d[0] = self.get_rloss(vfunc,depths_ar,annotate=True,ax=ax) 
                
                #===============================================================
                # #get average impacts for varios aggregation levels                
                #===============================================================
                for aggLevel in aggLevels_l:
                    log.info('%s aggLevel=%s'%(title, aggLevel))

                    #get aggregated depths
                    depMean_ar = np.array([a.mean() for a in np.array_split(depths_ar, math.ceil(len(depths_ar)/aggLevel))])
                    
                    #get these losses
                    res_d[aggLevel] = self.get_rloss(vfunc,depMean_ar, ax=ax, annotate=False, 
                                                       color=newColor_d[aggLevel], label=aggLevel)
 
 
                ax.legend() #add legend to scatter
                upd_xlim(ax)
                #===============================================================
                # box plot plotting
                #===============================================================
                ax = ax_d['rl_box'][xvar]
                
                #merge all rl results and clean
                rdf = pd.concat(res_d, keys=res_d.keys(), axis=1).drop(vfunc.xcn, axis=1, level=1).droplevel(level=1, axis=1)
                rd = {k:ser.dropna() for k,ser in rdf.items()}
                ax.boxplot(rd.values(), labels=['Agg=%i'%k for k in rd.keys()])
                ax.set_ylabel(ycn)
                #ax.legend()
                
                #===============================================================
                # wrap
                #===============================================================
                #change aggLevel keys
                #{'ag%i'%k:v for k,v in res_d.items()}
                res_lib[xmean][xvar] = res_d
                
            #===============================================================
            # wrap mean
            #===============================================================
            fig_lib[xmean] = fig
            log.info('finished xmean = %.2f'%xmean)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed on %i'%len(res_lib))
        
        #=======================================================================
        # #write the xmean plots---------
        #=======================================================================
        """
        plt.show()
        """
        if plot:
            for xmean, fig in fig_lib.items():
                log.info('handling figure %i'%fig.number)
                #harmonize axis
                self.apply_axd('set_xlim', ax_lib[xmean], rowNames=['depths_dist', 'rl_scatter'], 
                               left=self.master_xlim[0], right=self.master_xlim[1])
                
                #print(xmean)
                self.output_fig(fig, fname='%s_xmean%.2f'%(self.resname, xmean),
                                out_dir=os.path.join(self.out_dir, 'rl_mean_agg'))
            
            
        #=======================================================================
        # write meta
        #=======================================================================
        
 
        #=======================================================================
        # write results
        #=======================================================================
        #collapse intou 4 level dxcol
        rl_d = dict()
        for xmean, rd1 in res_lib.items():

            rl_d[xmean] = pd.concat({xvar:pd.concat(rd2, axis=1) for xvar, rd2 in rd1.items()}, axis=1)
            
        dxcol = pd.concat(rl_d, axis=1, names=['xmean', 'xvar', 'aggLevel', 'vars'])
        

 
        return dxcol
    
    def apply_axd(self, #apply a method to an axis within an ax_d 
                  methodName, 
                  ax_d,
                  rowNames=None, #default: apply to all
                  colNames=None,
                  **kwargs
                  ):
        #=======================================================================
        # #defaults
        #=======================================================================
        if rowNames is None:
            rowNames = list(ax_d.keys())
        
        
        if colNames is None:
            colNames =  list(ax_d[list(ax_d.keys())[0]].keys())
            
        #=======================================================================
        # check selection
        #=======================================================================
        l = set(rowNames).difference(ax_d.keys())
        assert len(l)==0, l
        
        l = set(colNames).difference(ax_d[list(ax_d.keys())[0]].keys())
        assert len(l)==0, l
        
        #=======================================================================
        # apply
        #=======================================================================
        res_lib = {k:dict() for k in ax_d.keys()}
        for rowName, ax_d0 in ax_d.items():
            if not rowName in rowNames: continue
 
            for colName, ax in ax_d0.items():
                if not colName in colNames: continue
                
                res_lib[rowName][colName] = getattr(ax, methodName)(**kwargs)
                
        return res_lib
                
 

    
    def get_rloss(self, 
                    vfunc,depths_ar, 
                    ax=None,
                    label='base',
                    annotate=False,
                    color='black',
                    linekwargs = dict(s=10, marker='x', alpha=0.5)
                    ):
                    #vfunc, ax_d, res_d, xvar, ax, depths_ar):
        
        rl_ar = vfunc.get_rloss(depths_ar)
 
        #=======================================================================
        # #plot these
        #=======================================================================
        if not ax is None:
            """
            plt.show()
            """
            ax.scatter(depths_ar, rl_ar,  color=color,label=label, **linekwargs)
            
            #label
            if annotate:
                #ax.legend()
                ax.set_title(vfunc.name)
                ax.set_ylabel(vfunc.ycn)
                ax.set_xlabel(vfunc.xcn)
        
 
        
        return pd.DataFrame([depths_ar, rl_ar], index=[vfunc.xcn, vfunc.ycn]).T.sort_values(vfunc.xcn).reset_index(drop=True) 
            
    def plot_rv(self, #get a plot of a scipy distribution
                rv,
                ax=None,
                figNumber = 1,
                color='red',
                xlab=None,
                title=None,
                lineKwargs = {
                    #'color':'red'
                    }
                ):
        
        #=======================================================================
        # setup data
        #=======================================================================
        #get x domain
        xar = np.linspace(rv.ppf(0.01),rv.ppf(0.99), 100)
        
        
        #=======================================================================
        # setup figure
        #=======================================================================
        if ax is None:
            ax = get_ax(figNumber=figNumber)

            
        #=======================================================================
        # #plot
        #=======================================================================
        #pdf
        ax.plot(xar, rv.pdf(xar), label='%s pdf'%rv.dist.name, color=color,
                 **lineKwargs)
        
        
        
        #=======================================================================
        # style
        #=======================================================================
        ax.grid()
        ax.legend()
        if not title is None: ax.set_title(title)
        if not xlab is None: ax.set_xlabel(xlab)
        ax.set_ylabel('frequency')
        
        d = {k: getattr(rv, k)() for k in ['mean', 'std', 'var']}
        
        text = '\n'.join(['%s:%.2f'%(k,v) for k,v in d.items()])
        anno_obj = ax.text(0.1, 0.9, text, transform=ax.transAxes, va='center',
                           fontsize=8, color=color)
        
 
        return ax
        """
        plt.show()
        """

 

    def get_joins(self, df_d, 
                  vidnm=None):
        
        if vidnm is None: vidnm=self.vidnm
        
        
        df1 = df_d['damage_function'].set_index(vidnm)
        
 
        
        
        #join some tabs
        meta_d = dict()
        link_colns = set() #columns for just linking
        for tabName in [
                        'damage_format', 
                        'function_format', 
                        'coverage', 
                        'sector',
                        'unicede_occupancy', #property use type    
                        'construction_material',
                        'building_quality',
                        'number_of_floors',
                        'basement_occurance',
                        'precaution',
                        #'con',
                        
                        #'d',
                        ]:
            #===================================================================
            # #get id frame
            #===================================================================
            jdf = df_d[tabName].copy()
            
            #index
            assert vidnm == jdf.columns[0]

 
            if not jdf[vidnm].is_unique:
                raise Error('non-unique indexer \'%s\' on \'%s\''%(vidnm, tabName))
            
            jdf = jdf.set_index(vidnm)
            
            link_colns.update(jdf.columns)
            #===================================================================
            # #add descriptions
            #===================================================================
            container_tabName = '%s_container'%tabName
            assert container_tabName in df_d, container_tabName
            container_jdf = df_d[container_tabName]
            
            container_jcoln = container_jdf.columns[0]
            assert container_jcoln in jdf.columns
            
            jdf1 = jdf.join(container_jdf.set_index(container_jcoln), on=container_jcoln)
            
            
            #===================================================================
            # join to main
            #===================================================================
            

 
            df1 = df1.join(jdf1, on=vidnm)
            
            #===================================================================
            # meta
            #===================================================================
            meta_d[tabName] = {'shape':str(jdf1.shape), 'jcoln':vidnm, 'columns':jdf1.columns.tolist(),
                               'container_tabn':container_tabName, 
                               'desc_colns':container_jdf.columns.tolist(),
                               'link_colns':jdf.columns.tolist()}
        
        #=======================================================================
        # drop link columns
        #=======================================================================
        df2 = df1.drop(link_colns, axis=1)
        
        
        return df2, pd.DataFrame.from_dict(meta_d).T
    
    
    def get_matrix_fig(self, #conveneince for getting a matrix plot with consistent object access
                       row_keys, #row labels for axis
                       col_keys, #column labels for axis
                       
                       fig_id=0,
                       figsize=None,
                        tight_layout=False,
                        constrained_layout=True,
                        
                        sharex=False, 
                         sharey='row',
                        
                       ):
        
        
        #=======================================================================
        # defautls
        #=======================================================================
        if figsize is None: figsize=self.figsize
        
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(row_keys, list)
        assert isinstance(col_keys, list)
        
        if fig_id in plt.get_fignums():
            plt.close()
        
        #=======================================================================
        # build figure
        #=======================================================================
        
        fig = plt.figure(fig_id,
            figsize=figsize,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout)
        
        # populate with subplots
        ax_ar = fig.subplots(nrows=len(row_keys), ncols=len(col_keys),
                             sharex=sharex, sharey=sharey,
                             )
        
        #convert to array
        if not isinstance(ax_ar, np.ndarray):
            assert len(row_keys)==len(col_keys)
            assert len(row_keys)==1
            
            ax_ar = np.array([ax_ar])
            
        
        #=======================================================================
        # convert to dictionary
        #=======================================================================
        ax_d = dict()
        for i, row_ar in enumerate(ax_ar.reshape(len(row_keys), len(col_keys))):
            ax_d[row_keys[i]]=dict()
            for j, ax in enumerate(row_ar.T):
                ax_d[row_keys[i]][col_keys[j]]=ax
                
            
 
            
        return fig, ax_d
    
    def get_plot_hndls(self,
                         df_raw,
                         coln = 'model_id', #style group
                         vidnm = None, #indexer for damage functions
                         colorMap = 'gist_rainbow',
                         ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if vidnm is None: vidnm=self.vidnm
        
       
        from matplotlib.lines import Line2D
        log = self.logger.getChild('plot_hndls')
        
        #=======================================================================
        # checks
        #=======================================================================
        assert len(df_raw)<=20, 'passed too many plots... this is over plotting'
        #=======================================================================
        # color by model_id
        #=======================================================================
        if df_raw[coln].isna().any():
            log.warning('got %i/%i nulls on \'%s\'... filling with mode'%(
                df_raw[coln].isna().sum(), len(df_raw), coln))
            
            df1 = df_raw.fillna(df_raw[coln].mode()[0])
        else:
            df1 = df_raw.copy()
            
            
        """
        view(df1)
        """
        res_df = pd.DataFrame(index=df1.index)
        #=======================================================================
        # #retrieve the color map
        #=======================================================================
        
        groups = df1.groupby(coln)
        
        cmap = plt.cm.get_cmap(name=colorMap)
        #get a dictionary of index:color values           
        d = {i:cmap(ni) for i, ni in enumerate(np.linspace(0, 1, len(groups)))}
        newColor_d = {i:matplotlib.colors.rgb2hex(tcolor) for i,tcolor in d.items()}
        
        #markers
        #np.repeat(Line2D.filled_markers, 3) 
        markers_d = Line2D.markers
        
        mlist = list(markers_d.keys())*3
        
        for i, (k, gdf) in enumerate(groups):
            
            #single color per model type
            res_df.loc[gdf.index.values, 'color'] = newColor_d[i]
            
 
            # marker within model_id
            if len(gdf)>len(markers_d):
                log.warning('\'%s\' more lines (%i) than markers!'%(k, len(gdf)))
 
 
            res_df.loc[gdf.index, 'marker'] = np.array(mlist[:len(gdf)])
             
            
 
        if not res_df.notna().all().all():
            log.warning('got %i nulls'%res_df.isna().sum().sum())
        
        
        #dummy for now
        return res_df.to_dict(orient='index')
    
    def write_dxcol(self, dxcol, dkey,
                    write_csv=True,
                    out_fp = None, 
                    logger=None):
        
        #=======================================================================
        # defautls
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('write_dxcol')
        assert isinstance(dxcol, pd.DataFrame)
        #=======================================================================
        # #picklle
        #=======================================================================
        if out_fp is None:
            out_fp = os.path.join(self.out_dir, '%s_%s.pickle' % (self.resname, dkey))
            
        else:
            write_csv=False
            
        with open(out_fp, 'wb') as f:
            pickle.dump(dxcol, f, pickle.HIGHEST_PROTOCOL)
        log.info('wrote %s to \n    %s' % (str(dxcol.shape), out_fp))
        
        
        #=======================================================================
        # #csv
        #=======================================================================
        if write_csv:
            out_fp2 = os.path.join(self.out_dir, '%s_%s.csv' % (self.resname, dkey))
            dxcol.to_csv(out_fp2, 
                index=True) #keep the names
            
            log.info('wrote %s to \n    %s' % (str(dxcol.shape), out_fp2))
        
        return out_fp
    
    def output_fig(self, 
                   fig,
                   
                   #file controls
                   out_dir = None, overwrite=None, 
                   out_fp=None, #defaults to figure name w/ a date stamp
                   fname = None, #filename
                   
                   #figure write controls
                 fmt='svg', 
                  transparent=True, 
                  dpi = 150,
                  logger=None,
                  ):
        #======================================================================
        # defaults
        #======================================================================
        if out_dir is None: out_dir = self.out_dir
        if overwrite is None: overwrite = self.overwrite
        if logger is None: logger=self.logger
        log = logger.getChild('output_fig')
        
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        #=======================================================================
        # precheck
        #=======================================================================
        
        assert isinstance(fig, matplotlib.figure.Figure)
        log.debug('on %s'%fig)
        #======================================================================
        # output
        #======================================================================
        if out_fp is None:
            #file setup
            if fname is None:
                try:
                    fname = fig._suptitle.get_text()
                except:
                    fname = self.name
                    
                fname =str('%s_%s'%(fname, self.resname)).replace(' ','')
                
            out_fp = os.path.join(out_dir, '%s.%s'%(fname, fmt))
            
        if os.path.exists(out_fp): 
            assert overwrite
            os.remove(out_fp)

            
        #write the file
        try: 
            fig.savefig(out_fp, dpi = dpi, format = fmt, transparent=transparent)
            log.info('saved figure to file:   %s'%out_fp)
        except Exception as e:
            raise Error('failed to write figure to file w/ \n    %s'%e)
        
        return out_fp
        
        
        
        
def run_plotVfunc( 
        tag='r0',
        
 
        #data selection
        #vid_l=[796],
        vid_l=None,
        gcoln = 'sector_attribute', #how to spilt figures
        style_gcoln = 'precaution_attribute',
        max_mod_cnt=None,
        
        
         selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, 
                              3, #flemo 
                              #4, 6, 7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
        
        #style
        figsize=(6,6),
        xlims=(0,2),
        ylims=(0,100),
        #=======================================================================
        # #debugging controls
        #=======================================================================
        #=======================================================================
 
        # 
        #by record count
        #debug_len=20,
        debug_len=None,
        # 
        # #use some preloaded data (saves lots of time during loading)
        # debug_fp=r'C:\LS\09_REPOS\02_JOBS\2107_obwb\outs\9vtag\r0\20211230\raw_9vtag_r0_1230.csv',
        # #debug_fp=None,
        #=======================================================================

        ):
 
    
 
    with Session(tag=tag,  overwrite=True,  name='plotAll',
                 # figsize=figsize,
                 ) as ses:
        
 
        
 
        vid_df = ses.get_data('vid_df',
                      vid_l=vid_l, max_mod_cnt=max_mod_cnt, selection_d=selection_d)
        
        if not debug_len is None:
            vid_df = vid_df.sample(debug_len)
        
 
        """
        view(vid_df)
        view(gdf)
        
        """
        
        for k, gdf in vid_df.groupby(gcoln):
            if not len(gdf)<=20:
                ses.logger.warning('%s got %i...clipping'%(k, len(gdf)))
                gdf = gdf.iloc[:20, :]
                                   
 
            phndl_d = ses.get_plot_hndls(gdf, coln=style_gcoln)
            
            fig = ses.plot_all(vid_l=phndl_d.keys(), phndl_d=phndl_d, vid_df=gdf,
                         figsize=figsize, xlims=xlims,ylims=ylims,
                         title='%s (%s) w/ %i'%(gdf.iloc[0,:]['abbreviation'], k, len(phndl_d)))
            
            ses.output_fig(fig, fname='%s_vfunc_%s'%(ses.resname, k))
            
            #clear vcunfs
            del ses.data_d['vf_d']
            
        
        out_dir = ses.out_dir
        
    return out_dir


def run_aggErr1(#agg error per function
        tag='r1',
        
        #selection
        vid_l=[
                796, #Budiyono (2015) 
               402, #MURL linear
               852, #Dutta (2003) nice and parabolic
               33, #FLEMO resi...
               332, #FLEMO commericial
               ], #running on a single function
        
        #run control
        dfp_d = {
            #'rlMeans_dxcol':r'C:\LS\09_REPOS\02_JOBS\2112_Agg\outs\aggErr1\r0\20220101\aggErr1_r0_0101_rlMeans_dxcol.pickle',
            #'rl_dxcol':r'C:\LS\09_REPOS\02_JOBS\2112_Agg\outs\aggErr1\r0\20220101\aggErr1_r0_0101_rl_dxcol.pickle',
            },
        
        #plot control
 
        plot_meanCalcs=False,
 
 
        
        ):
    """
    nice plots showing the average impact vs. depth of
        aggregated levels (color sequence by agg level)
        not aggregated (black)
        
    xmean will relate to the xaxis values
    
    separate axis for
        xvar (depths variance)
        
    separate figure for
        vfunc
            

    
    """
 
    
 
    with Session(tag=tag,  overwrite=True,  name='aggErr1',dfp_d=dfp_d,
                 # figsize=figsize,
                 ) as ses:
        
 
        ses.run_dd_agg(vid_l=vid_l, plot_meanCalcs=plot_meanCalcs,
 
                       )

 
        
        out_dir = ses.out_dir
        
    return out_dir



if __name__ == "__main__": 
    
    output = run_aggErr1()
    #output = run_plotVfunc()
    # reader()
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))