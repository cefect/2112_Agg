'''
Created on Dec. 31, 2021

@author: cefect

build functions from tables and plot
    let's use hp.coms, but not Canflood
'''


#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, pickle, weakref, inspect, copy
import pandas as pd
import numpy as np
import qgis.core

start = datetime.datetime.now()
print('start at %s' % start)

from hp.oop import Basic, Error
from hp.pd import view, get_bx_multiVal
 

label_conversion = {
    'relative_loss':'rl',
    }
    
    
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



class Vfunc(object):
    rlcn = 'rl'
    wdcn = 'wd'
    
    def __init__(self,
                 name='vfunc',
                 logger=None,
                 session=None,
                 meta_d = {}, #optional attributes to attach
                 ):
        
        #=======================================================================
        # attachments
        #=======================================================================
        self.name=name
        self.logger = logger.getChild(name)
        self.session=session
        self.meta_d=meta_d
        
    def set_ddf(self,
                df_raw,
                logger=None,
                ):
        
        #=======================================================================
        # defai;ts
        #=======================================================================
        if logger is None: logger=self.logger
        #log=logger.getChild('set_ddf')
        
        rlcn = self.rlcn
        wdcn = self.wdcn
        
        
        
        
        
        #precheck
        assert isinstance(df_raw, pd.DataFrame)
        l = set(df_raw.columns).symmetric_difference([rlcn, wdcn])
        assert len(l)==0, l
        
        
        df1 = df_raw.copy().sort_values(wdcn).loc[:, [wdcn, rlcn]]
        
        #check monotoniciy
        for coln, row in df1.items():
            assert np.all(np.diff(row.values)>=0), '%s got non mono-tonic %s vals \n %s'%(
                self.name, coln, df1)
        
        self.ddf = df1
        
        #log.info('attached %s'%str(df1.shape))
        
        return self
         
 

class Session(Basic):
    vidnm = 'df_id' #indexer for damage functions
    
    def __init__(self, 
                  work_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg',
                  mod_name = 'main.py',
                 **kwargs):
        
        super().__init__(work_dir=work_dir, mod_name=mod_name, 
                         
                         **kwargs)
        
        self.workers_d = dict()
        
    def load_db(self,
                fp=r'C:\LS\09_REPOS\02_JOBS\2112_Agg\figueiredo2018\cef\csv_dump.xls',
                ):
        log = self.logger.getChild('load_db')
        df_d = pd.read_excel(fp, sheet_name=None)
        
        log.info('loaded %i pages from \n    %s\n    %s'%(
            len(df_d), fp, list(df_d.keys())))
        
        
        self.df_d = df_d
        
        

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
    
    
    
    def select_dfuncs(self,
                      df_d = None,
                     selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[1, 2, 3, 4, 6, 7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
                     vidnm = None, #indexer for damage functions
                     ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('spawn_dfuncs')
        if df_d is None: df_d = self.df_d
        if vidnm is None: vidnm=self.vidnm
        meta_lib=dict()
        
        
        #=======================================================================
        # identify valid model_ids
        #=======================================================================
        #get metadata for selection
        df1, meta_lib['join_summary'] = self.get_joins(df_d)
        
        log.info('joined %i tabs to get %s' % (len(meta_lib['join_summary']), str(df1.shape)))
        
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
        
        """
        view(df2)
        view(df1.loc[bx, :])
        """
        
        return df2
        

    def spawn_dfuncs(self,
                     vid_df,
                     df_d = None,
 
                     
                     vidnm = None, #indexer for damage functions
                     ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('spawn_dfuncs')
        if df_d is None: df_d = self.df_d
        if vidnm is None: vidnm=self.vidnm
        meta_lib=dict()

        
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
                name=row['abbreviation']+'_%03i'%vid,
                logger=self.logger, session=self,
                meta_d = row.dropna().to_dict(), 
                ).set_ddf(ddf)
                
        log.info('spawned %i vfuncs'%len(vf_d))
        
        self.vf_d = vf_d
        
        return self.vf_d
            
 
    def build_plot_hndls(self,
                         df_raw,
                         vidnm = None, #indexer for damage functions
                         colorMap = 'gist_rainbow',
                         ):
        
        if vidnm is None: vidnm=self.vidnm
        
        res_df = pd.DataFrame(index=df_raw.index)
        from matplotlib.lines import Line2D
        
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
        
        
        for i, (k, gdf) in enumerate(groups):
            #single color per model type
            res_df.loc[gdf.index.values, 'color'] = newColor_d[i]
            
 
            # marker within model_id
 
 
            res_df.loc[gdf.index, 'marker'] = np.array(Line2D.filled_markers[:len(gdf)])
            
 
        assert res_df.notna().all().all()
        
        
        #dummy for now
        return res_df.to_dict(orient='index')
        
        """
        view(res_df.join(df_raw))
        view(df_raw)
        """
        
        
        
        
    def plot_all(self,
                 phndl_d=None, #optional plot handles
                 vf_d=None,
                 vidnm = None, #indexer for damage functions
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
            
            #get data
            ddf = vfunc.ddf
            xar, yar = ddf.T.values[0], ddf.T.values[1]
            
 
            lines = ax.plot(xar, yar, label=vfunc.name,  
                            **{**lineKwargs, **phndl_d[vid]},
                            )
            
        log.info('added %i lines'%len(vf_d))
        
        #=======================================================================
        # style
        #=======================================================================
        xvar, yvar = ddf.columns[0], ddf.columns[1]
        
        ax.set_xlabel('\'%s\' water depth (m)'%xvar)
        ax.set_ylabel('\'%s\' relative loss (pct)'%yvar)
        ax.grid()
 
        ax.set_title('\'%s\' vs \'%s\' on %i'%(xvar, yvar, len(vf_d)))
        
        #legend
        #fig.legend()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        #=======================================================================
        # write figure
        #=======================================================================
        
        
        
        
        
def run_plotAll(#first iteration. predictions for just use1
        tag='r0',
        
 
        
        #=======================================================================
        # #debugging controls
        #=======================================================================
        #=======================================================================
 
        # 
        #by record count
        debug_len=10,
        #debug_len=1000,
        # 
        # #use some preloaded data (saves lots of time during loading)
        # debug_fp=r'C:\LS\09_REPOS\02_JOBS\2107_obwb\outs\9vtag\r0\20211230\raw_9vtag_r0_1230.csv',
        # #debug_fp=None,
        #=======================================================================

        ):
 
    
 
    with Session(tag=tag,  overwrite=True,  
                 # figsize=figsize,
                 ) as ses:
        
        #=======================================================================
        # load data
        #=======================================================================
 
            
        ses.load_db()
        
 
        #=======================================================================
        # select models
        #=======================================================================
        vid_df = ses.select_dfuncs()
        
        if not debug_len is None:
            vid_df = vid_df.sample(debug_len)
        
        #=======================================================================
        # build vfuncs
        #=======================================================================
        ses.spawn_dfuncs(vid_df)
        phndl_d = ses.build_plot_hndls(vid_df)
        
        ses.plot_all(phndl_d=phndl_d)
        
        out_dir = ses.out_dir
        
    return out_dir
if __name__ == "__main__": 
    
    #output = run_use1()
    output = run_plotAll()
    # reader()
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))