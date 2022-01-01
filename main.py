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
import os, datetime, math, pickle
import pandas as pd
import numpy as np
import qgis.core

import scipy.stats 

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
        self.dd_ar = df1.values
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
        
        res_ar = np.apply_along_axis(lambda x:np.interp(x,
                                    dd_ar[0], #depths (xcoords)
                                    dd_ar[1], #damages (ycoords)
                                    left=0, #depth below range
                                    right=max(dd_ar[1]), #depth above range
                                    ),
                            0, xuq1)
        
 
        #=======================================================================
        # plug back in
        #=======================================================================
        """may be slower.. but easier to use pandas for the left join here"""
        rdf = rdf.join(
            pd.Series(res_ar, index=xuq1, name=self.ycn), on='wd_round')

 
 
        return rdf.sort_index()[self.ycn].values
    
    
    def plot(self,
             ax=None,
             figNumber=0,
             lineKwargs={},
             logger=None,
             ):
        
        
        #setup plot
        if ax is None:
            ax = get_ax(figNumber=figNumber)
            
        #get data
        ddf = self.ddf
        xar, yar = ddf.T.values[0], ddf.T.values[1]
        """
        plt.show()
        """
        
        
        return ax.plot(xar, yar, label=self.name, **lineKwargs)
            
                            
 
        
 
        
        
         
 

class Session(Basic):
    vidnm = 'df_id' #indexer for damage functions
    
    data_d = dict()
    
    ycn = 'rl'
    xcn = 'wd'
    
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
            
            'rlMeans_dxcol':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_rlMeans_dxcol(**kwargs),
                },
            
            'vid_df':{
                'build':lambda **kwargs:self.build_vid_df(**kwargs)
                
                },
            'vf_d':{
                'build':lambda **kwargs:self.build_vf_d(**kwargs)
                
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
        

    

            
    def build_rlMeans_dxcol(self,  #get aggregation erros for a single vfunc
                 vf_d=None,
                 vid_l=[],
                 dkey=None,
                 
                 #vid_df keys
                 max_mod_cnt=None,
                 
                 **kwargs
                 ):
        
        log = self.logger.getChild('r')
        assert dkey== 'rlMeans_dxcol'
 
        #=======================================================================
        # #load dfuncs
        #=======================================================================
        if vf_d is None:
            vf_d = self.get_data('vf_d', vid_l=vid_l)
 
            
 
             
            
 
         
        #======================================================================
        # calc for each
        #======================================================================
        res_lib = dict()
        for i, vfunc in vf_d.items():
            res_lib[vfunc.vid] = self.get_rl_mean_agg(vfunc, **kwargs)
            
            
        #=======================================================================
        # assemble
        #=======================================================================
        dxcol = pd.concat(res_lib, axis=1, 
                          names=[self.vidnm] + list(res_lib[vfunc.vid].columns.names))
        
        
        """
        dxcol.columns
        """
        #=======================================================================
        # write
        #=======================================================================
        """problems with csv, so also writing a pickel"""
        
        #picklle
        out_fp = os.path.join(self.out_dir, '%s_%s.pickle'%(self.resname, dkey))
        with open(out_fp,  'wb') as f:
            pickle.dump(dxcol, f, pickle.HIGHEST_PROTOCOL)
        log.info('wrote %s to \n    %s'%(str(dxcol.shape), out_fp))
        
        #csv
        out_fp = os.path.join(self.out_dir, '%s_%s.csv'%(self.resname, dkey))
        dxcol.to_csv(out_fp, 
                     index=True, #keep the names
                     )
        log.info('wrote %s to \n    %s'%(str(dxcol.shape), out_fp))
        
        """
        
        """
 
        
        return res_lib
    
 


        

    
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
        log = self.logger.getChild('build_vf_d')
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
                     vid_l = [],
 
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
            vid_df=self.get_data('vid_df', vid_l=vid_l)
            
        assert len(vid_l)==len(vid_df)
            
        if vidnm is None: vidnm=self.vidnm
 
            
        
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
                
        log.info('spawned %i vfuncs'%len(vf_d))
        
        #self.vf_d = vf_d
        
        return vf_d
            
 

 
    #===========================================================================
    # TOP RUNNERS---------
    #===========================================================================
    def analyze_rl_means(self,
                        dxcol_raw=None,
                        vf_d = None,
                        vid_l=[],
 
                        **kwargs
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('analyz')
        vidnm = self.vidnm
        #=======================================================================
        # load rloss per-mean results
        #=======================================================================
        if dxcol_raw is None:
            dxcol_raw = self.get_data('rlMeans_dxcol', vid_l=vid_l, plot=False)
            
            
        #check index
        mdex = dxcol_raw.columns
        assert np.array_equal(np.array(mdex.names), np.array([vidnm, 'xmean', 'xvar', 'aggLevel', 'vars']))
        mdex_names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}
            
        #check lvl0:vid
        res_vids_l = mdex.get_level_values(0).unique().tolist()
        l = set(vid_l).difference(res_vids_l)
        assert len(l) == 0, '%i requested vids not in the results: %s'%(len(l), l)
        
        #check lvl4:vars
        mdex.get_level_values(4).unique()
        

        
        #trim to selection
        dxcol_raw.loc[:, idx[vid_l, :,:]]
        
 
            
 
        if vf_d is None: 
            vf_d =  self.get_data('vf_d', vid_l=vid_l)
            
        log.info('on %s'%str(dxcol.shape))
        
        """
        vf_d.keys()
        """
        
 
        #=======================================================================
        # loop on xmean
        #=======================================================================
        for vid, gdf0 in dxcol.groupby(level=0, axis=1):
            vid = int(vid)
            log.info('on %i w/ %s'%(vid, str(gdf0.shape)))
            
            vfunc = vf_d[vid]
            #===================================================================
            # setup figure
            #===================================================================
            xvars_ar = np.array(gdf0.columns.get_level_values(2).astype(float).unique())
            fig, ax_d = self.get_matrix_fig(['dd'], xvars_ar.tolist(), figsize=(15,5))
            
            #===================================================================
            # draw the vfunc
            #===================================================================
            for xvar, ax in ax_d['dd'].items():
 
                vfunc.plot(ax=ax, lineKwargs = {'color':'black'})
            """
            plt.show()
            """
            #===================================================================
            # get agg level medians
            #===================================================================
            #gdf0.columns = gdf0.columns.remove_unused_levels()
            for xmean, gdf1 in gdf0.droplevel(0, axis=1).groupby(level=0, axis=1):
                for xvar, gdf2 in gdf1.droplevel(0, axis=1).groupby(level=0, axis=1):
                    xvar = float(xvar)
                    ax = ax_d['dd'][xvar]
                    for aggLevel, gdf3 in gdf2.droplevel(0, axis=1).groupby(level=0, axis=1):
                        """aggLevel is adding a decimal somewhere"""
                        pass
                        #ax.scatter(
            
 
    def plot_all(self,
                 phndl_d=None, #optional plot handles
                 vf_d=None,
                 vidnm = None, #indexer for damage functions
                 xlims = (0, 2),
                 lineKwargs = { #global kwargs
                     'linewidth':0.5
                     }
                 ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_all')
 
        if vidnm is None: vidnm=self.vidnm
        if vf_d is None: vf_d=self.vf_d
        
        log.info('on %i'%len(vf_d))
        
        if phndl_d is None:
            phndl_d = {k:{} for k in vf_d.keys()}
        #=======================================================================
        # setup figure
        #=======================================================================
        plt.close()
        fig = plt.figure(0,
            figsize=(10,10),
            tight_layout=False,
            constrained_layout=True)
        
        ax = fig.add_subplot(111)
        
        """
        plt.show()
        """
        
        
        
        for vid, vfunc in vf_d.items():
            log.info('on %s'%vfunc.name)
            
            vfunc.plot(ax=ax, lineKwargs = {**lineKwargs, **phndl_d[vid]}, logger=log)

            
        log.info('added %i lines'%len(vf_d))
        
        #=======================================================================
        # style
        #=======================================================================
        xvar, yvar = vfunc.xcn, vfunc.ycn
        
        if not xlims is None:
            ax.set_xlim(xlims)
        ax.set_xlabel('\'%s\' water depth (m)'%xvar)
        ax.set_ylabel('\'%s\' relative loss (pct)'%yvar)
        ax.grid()
 
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
        return self.output_fig(fig, fname=title, logger=log)
    
    #===========================================================================
    # HELPERS---------
    #===========================================================================

    def get_rl_mean_agg(self, #calculate rloss at different levels of aggregation
                          vfunc,
                          
                          xdomain=(0,2), #min/max of the xdomain
                          xdomain_res = 10, #number of increments for the xdomain
                          

                          
                          #random depths pramaeters
                          statsFunc = scipy.stats.norm, #function to use for the depths distribution
                          depths_resolution=20,  #number of depths to draw from the depths distritupion
                          
                          #plotting parameters
                        colorMap = 'cool',
                        plot=True,
                          ):

        #=======================================================================
        # defaults
        #=======================================================================
        
        log=self.logger.getChild('get_rl_mean')
        log.info('on %s'%vfunc.name)
        
        
        xvars_ar = np.linspace(.1,1,num=4)
        aggLevels_l= list(range(2,5, 1))
        
        fig_lib, meta_lib, res_lib = dict(), dict(), dict()
        
        #get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap)        
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(aggLevels_l, np.linspace(0, 1, len(aggLevels_l)))).items()}
        
 
 
        #=======================================================================
        # loop mean deths 
        #=======================================================================
        for xmean in np.linspace(xdomain[0],xdomain[1],num=xdomain_res):
            log.info('xmean=%.2f\n'%xmean)
            
            #===================================================================
            # setup working fig
            #===================================================================
            fig, ax_d = self.get_matrix_fig(['depths_dist', 'rl_scatter','rl_box'], xvars_ar.tolist(), figsize=(12,8))
            
            """
            plt.show()
            """
            
            #===================================================================
            # #depth variances (for generating depth values)
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
                ax.hist(depths_ar, color='blue', alpha=0.5, label='sample %i'%depths_resolution, density=True)
                ax.text(0.5, 0.9, '%i samples'%depths_resolution, transform=ax.transAxes, va='center',fontsize=8, color='blue')
                #===============================================================
                # #get average impact (w/o aggregation)
                #===============================================================
                ax = ax_d['rl_scatter'][xvar]
                res_d[0] = self.get_rloss(vfunc,depths_ar,annotate=True,ax=ax) 
                
                #get average impacts for varios aggregation levels                
                for aggLevel in aggLevels_l:
                    log.info('%s aggLevel=%s'%(title, aggLevel))
                    
 
                    #get aggregated depths
                    depMean_ar = np.array([a.mean() for a in np.array_split(depths_ar, math.ceil(len(depths_ar)/aggLevel))])
                    
                    #get these losses
                    res_d[aggLevel] = self.get_rloss(vfunc,depMean_ar, ax=ax, annotate=False, 
                                                       color=newColor_d[aggLevel], label=aggLevel)
 
 
                    
                    #get errors
                    pass
                
                #===============================================================
                # box plot plotting
                #===============================================================
                ax.legend() #add legend to scatter
                
                #box plots
 
                ax = ax_d['rl_box'][xvar]
                
                #merge all rl results and clean
                rdf = pd.concat(res_d, keys=res_d.keys(), axis=1).drop(vfunc.xcn, axis=1, level=1).droplevel(level=1, axis=1)
                rd = {k:ser.dropna() for k,ser in rdf.items()}
                ax.boxplot(rd.values(), labels=['Agg=%i'%k for k in rd.keys()])
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
            if plot:self.output_fig(fig, fname='%s_xmean%.2f'%(self.resname, xmean))
            log.info('finished xmean = %.2f'%xmean)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed on %i'%len(res_lib))
        
        #=======================================================================
        # #write the xmean plots
        #=======================================================================
        #=======================================================================
        # for xmean, fig in fig_lib.items():
        #     #print(xmean)
        #     self.output_fig(fig, fname='%s_xmean%.2f'%(self.resname, xmean))
        #=======================================================================
            
            
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
                         vidnm = None, #indexer for damage functions
                         colorMap = 'gist_rainbow',
                         ):
        
        if vidnm is None: vidnm=self.vidnm
        
        res_df = pd.DataFrame(index=df_raw.index)
        from matplotlib.lines import Line2D
        log = self.logger.getChild('plot_hndls')
        #=======================================================================
        # color by model_id
        #=======================================================================
        #res_d = {k:{} for k in df_raw.index} #start the container
        
        
        
        #retrieve the color map
        coln = 'model_id'
        groups = df_raw.groupby(coln)
        
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
             
            
 
        assert res_df.notna().all().all()
        
        
        #dummy for now
        return res_df.to_dict(orient='index')
        
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
        
        
        
        
def run_plotAll( 
        tag='r0',
        
 
        vid_l=[796],
        #vid_l=None,
        #=======================================================================
        # #debugging controls
        #=======================================================================
        #=======================================================================
 
        # 
        #by record count
        #debug_len=10,
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
        
 
        ses.build_df_d()
        
 
        vid_df = ses.build_vid_df(vid_l=vid_l)
        
        if not debug_len is None:
            vid_df = vid_df.sample(debug_len)
        
 
        ses.build_vf_d(vid_df)
        phndl_d = ses.get_plot_hndls(vid_df)
        
        ses.plot_all(phndl_d=phndl_d)
        
        out_dir = ses.out_dir
        
    return out_dir


def run_aggErr1(#agg error per function
        tag='r0',
 
        vid_l=[796], #running on a single function
        
        dfp_d = {
            'rlMeans_dxcol':r'C:\LS\09_REPOS\02_JOBS\2112_Agg\outs\aggErr1\r0\20220101\aggErr1_r0_0101_rlMeans_dxcol.pickle',
            }
        #=======================================================================
        # #debugging controls
        #=======================================================================
        #=======================================================================
 
        # 
        #by record count
        #debug_len=10,
 
        # 
        # #use some preloaded data (saves lots of time during loading)
        # debug_fp=r'C:\LS\09_REPOS\02_JOBS\2107_obwb\outs\9vtag\r0\20211230\raw_9vtag_r0_1230.csv',
        # #debug_fp=None,
        #=======================================================================
 
        
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
        
 
        ses.analyze_rl_means(vid_l=vid_l)

 
        
        out_dir = ses.out_dir
        
    return out_dir



if __name__ == "__main__": 
    
    output = run_aggErr1()
    #output = run_plotAll()
    # reader()
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))