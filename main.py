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
                 model_id=None, model_name='model_name',
                 name='vfunc',
                 logger=None,
                 session=None,
                 meta_d = {}, #optional attributes to attach
                 prec=4, #precsion for general rounding
                 ):
        
        #=======================================================================
        # attachments
        #=======================================================================
        self.model_id=model_id
        self.model_name=model_name
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
    
    data_d = dict() #datafiles loaded this session
    
    ofp_d = dict() #output filepaths generated this session
    
    ycn = 'rl'
    xcn = 'wd'
    colorMap = 'cool'
    
    bar_colors_d = {'total':'black', 'positives':'orange', 'negatives':'blue'}
    
    def __init__(self, 
                  work_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg',
                  mod_name = 'main.py',
                  dfp_d=dict(),
                  bk_lib=dict(),
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
            
            'rl_xmDisc_dxcol':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_rl_xmDisc_dxcol(**kwargs),
                },
            
            'rl_dxcol':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_rl_dxcol(**kwargs),
                },
            
            'model_metrics':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_model_metrics(**kwargs),
                },
 
            }
        
        self.bk_lib=bk_lib
        
        

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
            if dkey in self.bk_lib:
                bkwargs=self.bk_lib[dkey].copy()
                bkwargs.update(kwargs) #function kwargs take precident
                kwargs = bkwargs
                """
                clearer to the user
                also gives us more control for calls within calls
                """
                
            
            
            data = hndl_d['build'](dkey=dkey, **kwargs)
            
        #=======================================================================
        # store
        #=======================================================================
        self.data_d[dkey] = data
            
        log.info('finished on \'%s\' w/ %i'%(dkey, len(data)))
        
        return data
        

    
    def load_aggRes(self, #loading dxcol from pickels for rl_xmDisc_dxcol and rl_dxcol
                    fp):
        
        assert os.path.exists(fp), fp
        assert fp.endswith('.pickle')
        with open(fp, 'rb') as f: 
            dxcol_raw = pickle.load(f)
        
        #basic checks
        
        assert isinstance(dxcol_raw, pd.DataFrame), type(dxcol_raw)
        
        #check index
        mdex = dxcol_raw.columns
        """I guess only the level0 value is consistent"""
        for i, name in enumerate([self.vidnm]):
            assert name == mdex.names[i], 'bad mdex %i, %s !=%s'%(i, name, mdex.names[i])
        
        #=======================================================================
        # fix selection behavior
        #=======================================================================
        #swap default vid selection from random to explicit
        """
        while technically the user is making a bad request by random sampling while loading an intermediate result
            the expected behaior is to only apply random sampling during build
            
        leaving vid_l as passed so the user could request specific bids WITHIN hte loaded dxcol
        """
        if 'vid_df' in self.bk_lib:
            if 'vid_sample' in self.bk_lib['vid_df']:
                if not self.bk_lib['vid_df']['vid_sample'] is None:
                    self.logger.warning('replacing random with explicit smplaing on %s.%s'%('vid_df', 'vid_sample'))
                    assert self.bk_lib['vid_df']['vid_l'] is None, 'passing vid_df.vid_sample and vid_df.vid_l not allowed'
                    self.bk_lib['vid_df']['vid_l'] = dxcol_raw.columns.get_level_values(0).unique().tolist()
                    self.bk_lib['vid_df']['vid_sample'] = None
                
        
        
        #=======================================================================
        # check selection
        #=======================================================================
        #need to make sure the current selection matches whats loaded
        vid_df = self.get_data('vid_df')
        
        
        #check lvl0:vid
        vid_l = vid_df.index
        res_vids_l = mdex.get_level_values(0).unique().tolist()
        l = set(vid_l).difference(res_vids_l)
        assert len(l) == 0, '%i requested vids not in the results... are you random sampling?: \n%s'%(len(l), l)
        
        
        #=======================================================================
        # #trim to selection
        #=======================================================================
        dxcol = dxcol_raw.sort_index(axis=1)

        """may need to fix this for rl_dxcol"""
        dxcol = dxcol.loc[:, idx[vid_l, :,:]]
        mdex = dxcol.columns
        
        
        #check
        l = set(mdex.get_level_values(0).unique().tolist()).symmetric_difference(vid_l)
        assert len(l)==0
        
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
                      
                      #run control
                      write_model_summary=False,
                      
                      #simple vid selection
                      vid_l = None,
                      vid_sample=None,
                      
                      #attribute selection control
                      
                     selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[1, 2, 3, 4, 6, 7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
                     
                     max_mod_cnt = 10, #maximum dfuncs to allow from a single model_id
                     
                     
                     #keynames
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
        # attach data
        #=======================================================================
        #get metadata for selection
        df1, meta_lib['join_summary'] = self.get_joins(df_d)
        
        log.info('joined %i tabs to get %s' % (len(meta_lib['join_summary']), str(df1.shape)))
        

        
        #=======================================================================
        # basic selection-----------
        #=======================================================================
 
        #specivic fids
        if not vid_l is None:
            #check none o the other selectors are apssed
            assert vid_sample is None
            df3= df1.loc[vid_l, :]
        
 
        #=======================================================================
        # attribute selection-------
        #=======================================================================
        else:
            #id valid df_ids
            bx = get_bx_multiVal(df1, selection_d, log=log)
            
            df2 = df1.loc[bx, :] #.drop(list(selection_d.keys()), axis=1)
            
            log.info('selected %i/%i damageFunctions'%(len(df2), len(df1)))
            
     
            
            #=======================================================================
            # #by maximum model_id count
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
            
 
            
        #=======================================================================
        # random sampling within
        #=======================================================================
                #random vids
        if not vid_sample is None:
            res_df= df3.sample(vid_sample)
        else:
            res_df = df3
        #=======================================================================
        # checks
        #=======================================================================
        
        #check each of these have  ['wd', 'relative_loss']
        for tabName in  ['wd', 'relative_loss']:
            l = set(res_df.index).difference(df_d[tabName][vidnm])
            
            if len(l)>0:
                raise Error('requesting %i/%i %s w/o requisite tables\n    %s'%(
                    len(l), len(res_df), l))
            
        #=======================================================================
        # get a model summary sheet
        #=======================================================================
        if write_model_summary:
            coln = 'model_id'
            mdf = df1.drop_duplicates(coln).set_index(coln)
            
            
            mdf1 = mdf.join(df1[coln].value_counts().rename('vid_cnt')).dropna(how='all', axis=0)
            
            out_fp = os.path.join(self.out_dir, '%smodel_summary.csv'%self.resname)
            mdf1.to_csv(out_fp, index=True)
            
            log.info('wrote model summary to file: %s'%out_fp)
            
        
 
        
        
        
        """
        view(mdf1)
        view(df1)
        view(df3)
        view(df1.loc[bx, :])
        """
        log.info('finished on %s \n    %s'%(str(res_df.shape), res_df['abbreviation'].value_counts().to_dict()))
        return res_df
        

    def build_vf_d(self,
                     vid_df=None,
                     df_d = None,
                     
                     #vfuncs selection
 
 
                     #key names
                     dkey='vf_d',
                     vidnm = None, #indexer for damage functions
                     
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
            vid_df=self.get_data('vid_df')
            
 
        
            
        if vidnm is None: vidnm=self.vidnm
        xcn = self.xcn
        ycn = self.ycn
            
        
        #=======================================================================
        # spawn each df_id
        #=======================================================================
        def get_wContainer(tabName):
            df1 = df_d[tabName].copy()
            
            l = set(vid_df.index).difference(df1[vidnm])
            df1[vidnm].unique()
            if not len(l)==0:
                raise Error('missing %i keys in \'%s\''%(len(l), tabName))
            
 
            
            jdf = df_d[tabName+'_container'].copy()
            
            jcoln = jdf.columns[0]
            
            return df1.join(jdf.set_index(jcoln), on=jcoln) 
 
        lkp_d = {k:get_wContainer(k) for k in ['wd', 'relative_loss']}
 

        
        
        def get_by_vid(vid, k):
            try:
                df1 = lkp_d[k].groupby(vidnm).get_group(vid).drop(vidnm, axis=1).reset_index(drop=True)
            except Exception as e:
                raise Error(e)
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
                vid=vid, model_id=row['model_id'],model_name=row['abbreviation'],
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
    
    def build_rl_xmDisc_dxcol(self,  #get aggregation erros mean-discretized for all vfuncs
                 vf_d=None,
                 
                 #vfunc selection

                 
                 
                 #vid_df keys
                 dkey=None,
                 
                 chunk_size=1,
                 
                 #get_rl_xmDisc control
                 plot=True,
                 **kwargs
                 ):
        
        log = self.logger.getChild('r')
        assert dkey== 'rl_xmDisc_dxcol'
 
        #=======================================================================
        # #load dfuncs
        #=======================================================================
        if vf_d is None:
            vf_d = self.get_data('vf_d')
            
            
        #=======================================================================
        # setup
        #=======================================================================
        out_dir = os.path.join(self.out_dir, self.temp_dir, dkey)
        if not os.path.exists(out_dir): 
            os.makedirs(out_dir)
 
        #======================================================================
        # calc for each
        #======================================================================
 
        
        dump_fp_d, res_lib, master_vids, err_d = dict(), dict(), dict(), dict()
 
        for i, (vid, vfunc) in enumerate(vf_d.items()):
            log.info('%i/%i on %s'%(i+1, len(vf_d), vfunc.name))
            try:
                res_lib[vfunc.vid] = self.get_rl_xmDisc(vfunc,logger=log.getChild(str(vid)),plot=plot, **kwargs)
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
            
        self.ofp_d[dkey] = self.write_dxcol(dxcol, dkey, logger=log)
        
        #=======================================================================
        # error check
        #=======================================================================
        for k, msg in err_d.items():
            log.error(msg)
            
        if len(err_d)>0:
            raise Error('got %i errors'%len(err_d))
        
 
 
        return dxcol
    
 
    def build_rl_dxcol(self, #combine discretized xmean results onto single domain
                        dxcol=None,
                        vf_d = None,
                        vid_df=None,
                        
                        #vid selection
 
                        
                        #run control
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
        if dxcol is None:
            dxcol = self.get_data('rl_xmDisc_dxcol')
        

        
        #get vid_df for slicing
        if vid_df is None:
            vid_df = self.get_data('vid_df')
            
            
        
        vid_l = vid_df.index.tolist()
        
        #=======================================================================
        # check
        #=======================================================================
        #check index
        mdex = dxcol.columns
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}    
        
        assert np.array_equal(np.array(mdex.names), np.array([vidnm, 'xmean', 'xvar', 'aggLevel', 'vars']))
        
            
        #check lvl0:vid
        res_vids_l = mdex.get_level_values(0).unique().tolist()
        l = set(vid_df.index).symmetric_difference(res_vids_l)
        assert len(l) == 0, '%i requested vids not in the results... are you random sampling?: \n%s'%(len(l), l)
        
        #check lvl4:vars
        l = set(mdex.get_level_values(4).unique()).symmetric_difference([xcn, ycn])
        assert len(l)==0, l
        
        #check lvl3:aggLevel
        assert 'int' in mdex.get_level_values(3).unique().dtype.name
        

        """only relevant when loading from compiled... so do this there
        #=======================================================================
        # #trim to selection
        #=======================================================================
 
        """
        
        
        log.info('loaded  rl_xmDisc_dxcol w/ %s'%str(dxcol.shape))
        
        #=======================================================================
        # get vfuncs-----
        #=======================================================================
        if vf_d is None: 
            vf_d =  self.get_data('vf_d', vid_df = vid_df)
            
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
        self.ofp_d[dkey] = self.write_dxcol(rdxcol, dkey,  logger=log)
        
        log.info('finished w/ %s'%str(rdxcol.shape))
        return rdxcol


    def build_model_metrics(self, #integrate delta areas
                        dxcol=None, #depth-damage at different AggLevel and xvars
                        
 
                        dkey=None,
                        **kwargs
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild(dkey)
        assert dkey == 'model_metrics'
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.get_data('rl_dxcol')
            
        mdex = dxcol.columns
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)} 
        vid_l = mdex.get_level_values(0).unique().tolist()
        #=======================================================================
        # check
        #=======================================================================
        #check index
        
        assert np.array_equal(np.array(mdex.names), np.array([vidnm, 'xvar', 'aggLevel']))
        
        
        #log.info('loaded  rl_dxcol w/ %s'%str(dxcol.shape))
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
            
            res_d[vid] = self.calc_areas(gdf0.droplevel(0, axis=1), logger=log.getChild('v%i'%vid), **kwargs)
            
            
        log.info('got %i'%len(res_d))
        rdxcol = pd.concat(res_d, axis=1, names=names_d.keys())

        rdxcol.index.name='errArea_type'
        
        #=======================================================================
        # write
        #=======================================================================
        self.ofp_d[dkey] = self.write_dxcol(rdxcol, dkey, write_csv=True)
        
        return rdxcol
 

 
    #===========================================================================
    # TOP RUNNERS---------
    #===========================================================================
    
    def plot_eA_box(self, #integrate delta areas
                        dxcol=None, #depth-damage at different AggLevel and xvars
                        
                        g0_coln=None, #coln for subfolder division
                        g1_coln = 'errArea_type', #coln for dividing figures
                        grp_colns = ['model_id', 'sector_attribute'], #coln for xaxis division (matrix rows)
 
                        #dkey='rl_dxcol',
 
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
        if dxcol is None:
            dxcol = self.get_data('model_metrics')
            
            
        #=======================================================================
        # get macro stats
        #=======================================================================

        
        #move the vid to the index
        dx1 = dxcol.T.unstack(level=vidnm).T.swaplevel(axis=0).sort_index(level=0)
 
        
        #calc each
        d = dict()
        for stat in ['mean', 'min', 'max']:
            d[stat] = getattr(dx1, stat)()
            
        #add back
        #dx2 = dx1.append(pd.concat([pd.concat(d, axis=1).T], keys=['stats']))
        stats_df = pd.DataFrame.from_dict(d)
        
        
        
        #=======================================================================
        # get grouped stats-----
        #=======================================================================
        
        #move index to columns
        dx2 = dx1.unstack(level='errArea_type')
        names_d = {lvlName:i for i, lvlName in enumerate(dx2.columns.names)}
        
        #retrieve attribute info
        vid_df = self.get_data('vid_df')
        
        l = set(grp_colns).difference(vid_df.columns)
        assert len(l)==0, l
        
        
        #add dummy indexer
        if g0_coln is None:
            g0_coln = 'group0'
            vid_df[g0_coln] = True
        
        #=======================================================================
        # #get figure for each attribute of interest
        #=======================================================================
        for i0, (g0val, vid_gdf) in enumerate(vid_df.groupby(g0_coln)):
            
            #setup suubfolder
            if len(vid_gdf)==len(vid_df):
                out_dir = os.path.join(self.out_dir, 'plot_eA_box')
            else:
                out_dir = os.path.join(self.out_dir, 'plot_eA_box', str(g0val))
 
 
                                              

        
            for i, coln in enumerate(grp_colns + ['all']):
                plt.close('all')
    
     
                #get grouping info
                if coln =='all':
                    gser = pd.Series(True, name='all', index=vid_gdf.index)
                else:
                    gser = vid_gdf[coln]
                
                for gval, gdxcol in dx2.groupby(g1_coln, axis=1):
                    """need this extra divisor"""
     
                    #build the matrix fig
                    fig = self.get_eA_fig(gdxcol.droplevel(names_d[g1_coln], axis=1),
                                         gser, fig_id=i, 
                                        logger=log.getChild(coln))
                    
                    #post
                    fig.suptitle(gval)
                    
                    
                    self.output_fig(fig, out_dir=out_dir, fname='%s_%s_%s'%(self.resname, coln, gval))
                
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
                
                
        
 
        
        return  
        """
        view(rdxcol)
        """
        
    def plot_eA_bars(self, #integrate delta areas
                        dxcol=None, #depth-damage at different AggLevel and xvars
                        
 
                        
                        grp_colns_fig = [ #pulling from vid_df
                            'model_id', 'sector_attribute'
                            ],
 
                        #dkey='rl_dxcol',
                        **kwargs
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('pBars')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.get_data('model_metrics')
            
        vid_df = self.get_data('vid_df')
        
        """
        view(vid_df)
        """
            
        #=======================================================================
        # data prep
        #=======================================================================

        
        #move the vid to the index
        dx1 = dxcol.T.unstack(level=vidnm).T.swaplevel(axis=0).sort_index(level=0)
        
        
        #move index to columns
        dx2 = dx1.unstack(level='errArea_type')
        
        
        #check
        l = set(grp_colns_fig).difference(vid_df.columns)
        assert len(l)==0, l
        
        #=======================================================================
        # join in grouping values
        #=======================================================================
        dx3 = dx2.copy()
        
        #build new index
        dfi = pd.Series(dx3.index).to_frame().join(vid_df.loc[:, grp_colns_fig], on=vidnm)
        dx3.index = pd.MultiIndex.from_frame(dfi)
        
        inames_d= {lvlName:i for i, lvlName in enumerate(dx3.index.names)}
        
        #=======================================================================
        # loop each fig group
        #=======================================================================
        for i, (gvals, gdx) in enumerate(dx3.groupby(level=[inames_d[k] for k in grp_colns_fig])):
            gk_d = dict(zip(grp_colns_fig, gvals))
            gv_str = '_'.join([str(e) for e in gvals])
        
            
            #drop extra levels
            dxcol_i = gdx.droplevel([inames_d[k] for k in grp_colns_fig])
            
            #plot
            plt.close('all')
            fig = self.get_eA_bars_fig(dxcol_i, fig_id=i, logger=log.getChild(gv_str), **kwargs)
            
            #post
            fig.suptitle(gv_str)
            
            
            self.output_fig(fig, out_dir=os.path.join(self.out_dir, 'plot_eA_bars'), 
                            fname='%s_%s'%(self.resname,gv_str))
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        
        
    def plot_xmDisc(self,
                    dxcol=None,
                    vf_d = None,
 
                        
                        #vid selection
 
                        
                        #run control
 
                        **kwargs
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_xmDisc')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.get_data('rl_xmDisc_dxcol')
            
        if vf_d is None:
            vf_d = self.get_data('vf_d')
        

        log.info('on %s'%str(dxcol.shape))
 
        #=======================================================================
        #  loop each dataset
        #=======================================================================
        names_d = {lvlName:i for i, lvlName in enumerate(dxcol.columns.names)}
        
        
        for i, ((vid, xmean), gdf) in enumerate(
            dxcol.groupby(level=[names_d[e] for e in ['df_id', 'xmean']], axis=1)):
            
 
            plt.close('all')
            fig = self.get_agg_imp_fig(gdf.droplevel([names_d[e] for e in ['df_id', 'xmean']], axis=1),
                                       vfunc=vf_d[vid],
                                       fig_id=0)
            
            

        
        
 
    def plot_all_vfuncs(self,
                 #data selection

                 
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
        log = self.logger.getChild('plot_all_vfuncs')
 
        vidnm=self.vidnm
        
        if vf_d is None: 
            vf_d=self.get_data('vf_d', vid_df=vid_df)
        
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
    def get_eA_bars_fig(self,
                        dxcol,
                        col_coln='xvar', #matrix plot rows
                        row_coln='aggLevel',#matrix plot columns
 
                        #plot style
                        colorMap=None,
                        bar_colors_d=None,
                        
                        fig_id=None,
                        ylims = (-30,30),
                        
                        #run control
                        logger=None,
                        ):
            
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('get_eA_bars_fig')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        if colorMap is None: colorMap=self.colorMap
        if bar_colors_d is None: bar_colors_d=self.bar_colors_d
        #=======================================================================
        # retrieve meta
        #=======================================================================
        mdex = dxcol.columns
        cnames_d= {lvlName:i for i, lvlName in enumerate(mdex.names)}
        
        xvar_l =  mdex.get_level_values(cnames_d[col_coln]).unique().tolist()
        rows_l = mdex.get_level_values(cnames_d[row_coln]).unique().tolist()
        
        
        #=======================================================================
        # #setup figure
        #=======================================================================
        
        fig, ax_d = self.get_matrix_fig(rows_l,xvar_l, 
                                        figsize=( len(xvar_l)*4, len(rows_l)*4),
                                        constrained_layout=True,
                                        fig_id=fig_id)
        
        #fig.suptitle(ser.name)
        """
        fig.show()
        """
        
        #get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(rows_l, np.linspace(0, 1, len(rows_l)))).items()}
             
        
        #=======================================================================
        # populate each axis
        #=======================================================================       
        for gvals, gdf in dxcol.groupby(level=[cnames_d[row_coln],cnames_d[col_coln]],  axis=1):
            #===================================================================
            # setup this group
            #===================================================================
            gk_d = dict(zip([row_coln,col_coln], gvals))
            gv_str = '_'.join([str(e) for e in gvals])
            
            """transforming to match other funcs"""
            df = gdf.droplevel([cnames_d[row_coln],cnames_d[col_coln]], axis=1).T
            
 
            #===================================================================
            # setup axis
            #===================================================================
            ax = ax_d[gvals[0]][gvals[1]]
            ax.set_title(', '.join(['%s=%s'%(k,v) for k,v in gk_d.items()]))
            
            
 
            #===================================================================
            # add bars per xgroupn
            #===================================================================
            locs = np.linspace(.1,.9,len(df.columns))
            width = 0.1/len(locs)
            
            """
            ax.clear()
            """
            
            for i, (label, row) in enumerate(df.iterrows()):
                ax.bar(locs+i*width, row.values, label=label, width=width, alpha=0.5,
                       color=bar_colors_d[label],
                       tick_label=row.index)
                
        
        #=======================================================================
        # post axisforamtting
        #=======================================================================
        firstRow = True
        for rowVal, d in ax_d.items():
            
            firstCol =True
            for colVal, ax in d.items():
 
                
                
                
                if firstRow:
                    ax.set_xlabel(df.columns.name)
                    
                    
                    
                    if firstCol: ax.legend()
                        
 
                    
                if firstCol:
                    ax.set_ylabel('area under rl difference curve')
                    ax.set_ylim(ylims)
                    
                    firstCol = False
            
            #wrap row
            if firstRow: 
                firstRow=False
            
        log.info('finished')
        return fig
                
            
  
    
    def get_eA_fig(self, #get grouped (gser) bar plots per aggLevel + xvar 
            dxcol, ser,
            row_coln = 'aggLevel', #how to divide matrix plot rows
            logger=None,
            
            #plot control
            ylims = (-30,30),
            fig_id=None,
 
            colorMap=None,
            ):
        """
        highlight
            aggLevel
            gropuVal
        
        matrix plot of boxplots
            columns: xvar
            rows:eatypes_l
                axis groups: gvals
            
        plt.close('all')
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('get_eA_fig')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        if colorMap is None: colorMap=self.colorMap
        #=======================================================================
        # retrieve meta
        #=======================================================================
        mdex = dxcol.columns
        cnames_d= {lvlName:i for i, lvlName in enumerate(mdex.names)}
        
        xvar_l = mdex.get_level_values(0).unique().tolist()
        rows_l = mdex.get_level_values(cnames_d[row_coln]).unique().tolist()
        
        
        #=======================================================================
        # #setup figure
        #=======================================================================
        
        fig, ax_d = self.get_matrix_fig(rows_l,xvar_l, 
                                        figsize=( len(xvar_l)*4, len(rows_l)*4),
                                        constrained_layout=True,
                                        fig_id=fig_id)
        
        #fig.suptitle(ser.name)
        
        #get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(rows_l, np.linspace(0, 1, len(rows_l)))).items()}
        

            
        #=======================================================================
        # join in grouping values
        #=======================================================================
        dxcol1 = dxcol.copy()
        
        
        dxcol1.index = pd.MultiIndex.from_frame(pd.Series(dxcol.index).to_frame().join(ser.rename('group'), on=vidnm))
        inames_d= {lvlName:i for i, lvlName in enumerate(dxcol1.index.names)}
        
        """
        view(dxcol1)
        ax.clear()
        fig.show()
        """
        
 
        for i,  (rowVal, gdx0) in enumerate(dxcol1.groupby(level=cnames_d[row_coln], axis=1)):
 
            
            #xvar loop
            for xvar, gdx1  in  gdx0.groupby(level=0, axis=1):
                
                #===============================================================
                # data prep
                #===============================================================
                #===============================================================
                # #get just this series
                # dser = gdx1.droplevel(level=0).droplevel(level=0, axis=1).iloc[:, 0].rename('vals')
                # 
                # #pivot out into frame (one column per group)
                # pdf = dser.to_frame().reset_index().pivot(columns='group')
 
                #===============================================================
                
                gd = {k:v.values.reshape(-1) for k,v in gdx1.groupby(level=inames_d['group'], axis=0)}
                #===============================================================
                # plotting prep
                #===============================================================
                ax = ax_d[rowVal][xvar]
                ax.set_title('%s=%s, xvar=%.2f'%(row_coln, rowVal, xvar))
                ax.set_xlabel(ser.name)
                ax.set_ylabel(ycn)

                #===============================================================
                # add the boxplot
                #===============================================================
                ax.boxplot(gd.values(), labels=gd.keys(),meanline=True,
                           boxprops={'color':newColor_d[rowVal]}, whiskerprops={'color':newColor_d[rowVal]})
                
                
                ax.set_ylim(ylims)
                
                #add the xaxis line
                
                ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], color='black', linewidth=0.5, linestyle='dashed')
                
 
                
        #=======================================================================
        # wrap
        #=======================================================================
 
                
        return fig
    
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
                         'dd':(0,80), 'delta':(-15,15), 'bars':(-15,15)
                         },
                     
                     xlims = (0,2),
 
                     
                     #plot style
                     colorMap=None,
                     prec=4, #rounding table values
                     title=None,
                     
                     #run control
                     out_dir=None,
                     
                     ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('calc_rl_v%s'%vfunc.vid)
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        if colorMap is None: colorMap=self.colorMap
        
        #output dir
        if out_dir is None: 
            out_dir=os.path.join(self.out_dir, 'models',str(vfunc.model_id), str(vfunc.vid))
        
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
 
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
        
        

        plt.close('all')
        fig, ax_d = self.get_matrix_fig(list(ylims_d.keys()), lVals_d['xvar'].tolist(), 
                                        figsize=(len(lVals_d['xvar'])*4,3*4),
                                        constrained_layout=True)
        
 
        if title is None:
            title = vfunc.name
        fig.suptitle(title)
 
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
                
                area_d['positives'] = scipy.integrate.trapezoid(diff1_ar, xar)
                
                #===============================================================
                # negative  areas
                #===============================================================
                #clear out negatives
                diff2_ar = diff_ar.copy()
                diff2_ar[diff_ar>=0] = 0
                
                area_d['negatives'] = scipy.integrate.trapezoid(diff2_ar, xar)
                
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
            
            locs = np.linspace(.1,.9,len(adf.columns))
            for i, (label, row) in enumerate(adf.iterrows()):
                ax.bar(locs+i*0.1, row.values, label=label, width=0.1, alpha=0.5,
                       color=self.bar_colors_d[label],
                       tick_label=row.index)
                
            
            
            #first row
            if xvar == min(lVals_d['xvar']):
                ax.set_ylabel('area under rl difference curve')
                ax.legend()
            
 
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
        
        self.output_fig(fig, fname='%s_%s_rlMeans'%(self.resname, vfunc.vid), out_dir=out_dir)
        
        return res_dxcol
                
 


    def calc_agg_imp(self, vfunc, 
                        
                        xmean, 
                        
                          #aggrevation levels to consider
                          aggLevels_l= [2, 
                                        #5, 
                                        100,
                                        ],
                        
                          #random depths pramaeters
                          xvars_ar = np.linspace(.1,1,num=2), #varice values to iterate over
                        statsFunc = scipy.stats.norm, #function to use for the depths distribution
                         depths_resolution=2000,  #number of depths to draw from the depths distritupion
                        
 
                        logger=None,
 
                        ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('calc_agg_imp')
        log.info('on %s'%vfunc.name)
        
 
        #===================================================================
        # #depth variances (for generating depth values)-------
        #===================================================================
        res_lib = dict()
        for xvar in xvars_ar:
            res_d = dict()
            title = 'xmean=%.2f xvar=%.2f' % (xmean, xvar)
            log.debug(title)
            #===============================================================
            # get depths
            #===============================================================
            #build depths distribution
 
            rv = statsFunc(loc=xmean, scale=xvar)
            #self.plot_rv(rv, ax=ax, xlab='depth (m)', title=title)
            #get depths set
            depths_ar = rv.rvs(size=depths_resolution)
 
            #===============================================================
            # #get average impact (w/o aggregation)
            #===============================================================
 
            res_d[0] = self.get_rloss(vfunc, depths_ar, annotate=True, ax=None)
            #===============================================================
            # #get average impacts for varios aggregation levels
            #===============================================================
            for aggLevel in aggLevels_l:
                log.debug('%s aggLevel=%s' % (title, aggLevel))
                #get aggregated depths
                depMean_ar = np.array([a.mean() for a in np.array_split(depths_ar, math.ceil(len(depths_ar) / aggLevel))])
                
                #get these losses
                res_d[aggLevel] = self.get_rloss(vfunc, depMean_ar, ax=None)
            
 
 
            
            #===================================================================
            # wrap
            #===================================================================
            res_lib[xvar] = res_d
        
        return res_lib 
    
    def get_agg_imp_fig(self, 
                        dxcol,
                        vfunc=None,
                        colorMap=None,
                        logger=None,
                        fig_id=0,
                        
                         statsFunc = scipy.stats.norm, #function to use for the depths distribution
                        ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_agg_imp_fig')
        log.info('on %s'%vfunc.name)
        
        if colorMap is None: colorMap=self.colorMap
        xcn, ycn = self.xcn, self.ycn
        
        names_d = {lvlName:i for i, lvlName in enumerate(dxcol.columns.names)}
        #=======================================================================
        # retrieve
        #=======================================================================
        aggLevels_l = dxcol.columns.get_level_values(names_d['aggLevel']).unique().tolist()
        xvars_ar = dxcol.columns.get_level_values(names_d['xvar']).unique()
        
        
        
        
        #get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap)        
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(aggLevels_l, np.linspace(0, 1, len(aggLevels_l)))).items()}
        
        #=================================================================== 
        # setup working fig 
        #===================================================================
        fig, ax_d = self.get_matrix_fig(['depths_dist', 'rl_scatter', 'rl_box'], xvars_ar.tolist(), 
            figsize=(4 * len(xvars_ar), 8), fig_id=fig_id)
        
        def upd_xlim(ax):
            new = ax.get_xlim()
            
            old = copy.copy(self.master_xlim)
            self.master_xlim = (
                                round(min(old[0], new[0]), 2),
                               round(max(old[1], new[1]), 2)
                               )
 
        #===================================================================
        # #depth variances (for generating depth values)-------
        #===================================================================
        res_lib = dict()
        for xvar in xvars_ar:
            res_d = dict()
            title = 'xmean=%.2f xvar=%.2f' % (xmean, xvar)
            log.debug(title)
            #===============================================================
            # get depths
            #===============================================================
            raise Error('need to store the depths_ar during get_rl_xmDisc so we can plot it here')
            #build depths distribution
            ax = ax_d['depths_dist'][xvar]
            rv = statsFunc(loc=xmean, scale=xvar)
            self.plot_rv(rv, ax=ax, xlab='depth (m)', title=title)
            #get depths set
            depths_ar = rv.rvs(size=depths_resolution)
            ax.hist(depths_ar, color='blue', alpha=0.5, label='sample %i' % depths_resolution, density=True, bins=20)
            ax.text(0.5, 0.9, '%i samples' % depths_resolution, transform=ax.transAxes, va='center', fontsize=8, color='blue')
            upd_xlim(ax)
            #===============================================================
            # #get average impact (w/o aggregation)
            #===============================================================
            ax = ax_d['rl_scatter'][xvar]
            res_d[0] = self.get_rloss(vfunc, depths_ar, annotate=True, ax=ax)
            #===============================================================
            # #get average impacts for varios aggregation levels
            #===============================================================
            for aggLevel in aggLevels_l:
                log.debug('%s aggLevel=%s' % (title, aggLevel))
            #get aggregated depths
                depMean_ar = np.array([a.mean() for a in np.array_split(depths_ar, math.ceil(len(depths_ar) / aggLevel))])
            #get these losses
                res_d[aggLevel] = self.get_rloss(vfunc, depMean_ar, ax=ax, annotate=False, 
                    color=newColor_d[aggLevel], label=aggLevel)
            
            ax.legend() #add legend to scatter
            upd_xlim(ax)
            #===============================================================
            # box plot plotting
            #===============================================================
            ax = ax_d['rl_box'][xvar]
            #merge all rl results and clean
            rdf = pd.concat(res_d, keys=res_d.keys(), axis=1).drop(vfunc.xcn, axis=1, level=1).droplevel(level=1, axis=1)
            rd = {k:ser.dropna() for (k, ser) in rdf.items()}
            ax.boxplot(rd.values(), labels=['Agg=%i' % k for k in rd.keys()])
            ax.set_ylabel(ycn)
            
            #===================================================================
            # wrap
            #===================================================================
            res_lib[xvar] = res_d
        
        return res_lib, fig

    def get_rl_xmDisc(self, #calculate rloss at different levels of aggregation
                          vfunc,
                          
 
                          #xvalues to iterate (xmean)
                          xdomain=(0,2), #min/max of the xdomain
                          xdomain_res = 5, #number of increments for the xdomain
 
                          
                          #plotting parameters
 
                        plot=False,
                        out_dir=None,
                        logger=None,
                          ):

        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_rl_xmDisc')
        log.info('on %s'%vfunc.name)
        
 
        xcn, ycn = self.xcn, self.ycn
        
        if out_dir is None: out_dir = os.path.join(self.out_dir, vfunc.name, 'xmean_disc')
        
        fig_lib, meta_lib, res_lib, ax_lib = dict(), dict(), dict(), dict()
        

        
 
        #=======================================================================
        # #xaxis control
        #=======================================================================
        self.master_xlim = (0,0) #set for controlling bounds
        

            
        
        #=======================================================================
        # loop mean deths 
        #=======================================================================
 
        xmean_ar = np.linspace(xdomain[0],xdomain[1],num=xdomain_res)
        for i, xmean in enumerate(xmean_ar):
            log.info('%i/%i xmean=%.2f'%(i, len(xmean_ar), xmean))
            
 
            
            res_lib[xmean]= self.calc_agg_imp(vfunc,xmean, 
                                                    logger=log.getChild(str(i)))
 
                
            #===============================================================
            # wrap mean
            #===============================================================
 
            log.debug('finished xmean = %.2f'%xmean)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.debug('finsihed on %i'%len(res_lib))
        
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
                                fmt='png',
                                out_dir=out_dir)
            
            
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
                    write_csv=False,
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
        log.info('wrote \'%s\' (%s) to \n    %s' % (dkey, str(dxcol.shape), out_fp))
        
        
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
    
    def __exit__(self, #destructor
                 *args, **kwargs):
        
        print('Session.__exit__ on \'%s\''%self.__class__.__name__)
        
        #=======================================================================
        # log major containers
        #=======================================================================
        print('__exit__ w/ data_d.keys(): %s'%(list(self.data_d.keys())))
        
        if len(self.ofp_d)>0:
            print('__exit__ with %i ofp_d:'%len(self.ofp_d))
            for k,v in self.ofp_d.items():print('    \'%s\':r\'%s\','%(k,v))
              
              
        
        
        super().__exit__(*args, **kwargs)
        
        
        
        
def run_plotVfunc( 
        tag='r1',
        
 
        #data selection
        #vid_l=[796],
        vid_l=None,
        gcoln = 'model_id', #how to spilt figures
        style_gcoln = 'sector_attribute',
        max_mod_cnt=10,
        vid_sample=10,
        
         selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              1, 2, 
                              3, #flemo 
                              4, 6, 7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
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

        # 
        # #use some preloaded data (saves lots of time during loading)
        # debug_fp=r'C:\LS\09_REPOS\02_JOBS\2107_obwb\outs\9vtag\r0\20211230\raw_9vtag_r0_1230.csv',
        # #debug_fp=None,
        #=======================================================================

        ):
 
    
 
    with Session(tag=tag,  overwrite=True,  name='plotAll',
                 bk_lib = {
                     'vid_df':dict(
                      vid_l=vid_l, max_mod_cnt=max_mod_cnt, selection_d=selection_d, vid_sample=vid_sample
                                    ),
                     }
                 ) as ses:
        
 
        
 
        vid_df = ses.get_data('vid_df')
 
 
        """
        view(vid_df)
        view(gdf)
        
        """
        
        for k, gdf in vid_df.groupby(gcoln):
            if not len(gdf)<=20:
                ses.logger.warning('%s got %i...clipping'%(k, len(gdf)))
                gdf = gdf.iloc[:20, :]
                                   
 
            phndl_d = ses.get_plot_hndls(gdf, coln=style_gcoln)
            
            fig = ses.plot_all_vfuncs(phndl_d=phndl_d, vid_df=gdf,
                         figsize=figsize, xlims=xlims,ylims=ylims,
                         title='%s (%s) w/ %i'%(gdf.iloc[0,:]['abbreviation'], k, len(phndl_d)))
            
            ses.output_fig(fig, fname='%s_vfunc_%s'%(ses.resname, k))
            
            #clear vcunfs
            del ses.data_d['vf_d']
            
        
        out_dir = ses.out_dir
        
    return out_dir


def run_aggErr1(#agg error per function
        tag='r2',
        
        #selection
        #=======================================================================
        # vid_l=[
        #         796, #Budiyono (2015) 
        #        #402, #MURL linear
        #        #852, #Dutta (2003) nice and parabolic
        #        #33, #FLEMO resi...
        #        #332, #FLEMO commericial
        #        ], #running on a single function
        #=======================================================================
        vid_l=None,
        
        vid_sample=None,
        max_mod_cnt=3,
        
         selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              3, #flemo 
                              4, 6, 
                              #7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
 
        #run control
        dfp_d = {
            'rl_xmDisc_dxcol':r'C:\LS\09_REPOS\02_JOBS\2112_Agg\outs\aggErr1\r2\20220103\aggErr1_r2_0103_rl_xmDisc_dxcol.pickle',
         
            },
        
        #plot control
 
 
 
        
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
                 bk_lib = {
                     'vid_df':dict(
                            selection_d=selection_d,vid_l = vid_l,vid_sample=vid_sample,max_mod_cnt=max_mod_cnt,
                                    ),
                     'rl_xmDisc_dxcol':dict(plot=False),
                            }
                 # figsize=figsize,
                 ) as ses:
        
        vid_df=ses.get_data('vid_df')
        
        #discretization calcs
        ses.plot_xmDisc()
        
        #combined box plots
        ses.plot_eA_box(grp_colns = ['model_id', 'sector_attribute'])
        
        #per-model bar plots
        ses.plot_eA_bars()
        
        
 
         
        
        

 
        
        out_dir = ses.out_dir
        
    return out_dir



if __name__ == "__main__": 
    
    output = run_aggErr1()
    #output = run_plotVfunc()
    # reader()
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))