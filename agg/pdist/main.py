'''
Created on Dec. 31, 2021

@author: cefect

explore errors in impact estimates as a result of aggregation using pdist generated depths
    let's use hp.coms, but not Canflood
    using damage function csvs from figurero2018 (which were pulled from a db)
    
trying a new system for intermediate results data sets
    key each intermediate result with corresponding functions for retriving the data
        build: calculate this data from scratch (and other intermediate data sets)
        compiled: load straight from HD (faster)
        
    in this way, you only need to call the top-level 'build' function for the data set you want
        it should trigger loads on lower results (build or compiled depending on what filepaths have been passed)
        
        
TODO: migrate to new oop
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
matplotlib.rcParams['figure.titlesize']=12
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)

#===============================================================================
# custom imports
#===============================================================================
from scripts import Session
        
        
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
        # debug_fp=r'C:\LS\10_OUT\2107_obwb\outs\9vtag\r0\20211230\raw_9vtag_r0_1230.csv',
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
        
 
        
 
        vid_df = ses.retriee('vid_df')
 
 
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

        
        #selection

        vid_l=None,
        
        vid_sample=None,
        max_mod_cnt=None,
        
         selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              3, #flemo 
                              4, 6, 
                              7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
         
         #run control
         rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 10, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             #5, 
                             10,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.5,1,num=2), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=500,  #number of depths to draw from the depths distritupion
                          ),
         plot_rlMeans = True,
         overwrite=True,
 
 
        
        **kwargs):
 
    
 
    with Session(overwrite=overwrite,  
                 bk_lib = {
                     'vid_df':dict(
                            selection_d=selection_d,vid_l = vid_l,vid_sample=vid_sample,max_mod_cnt=max_mod_cnt,
                                    ),
                     'rl_xmDisc_dxcol':rl_xmDisc_dxcol_d,
                     'rl_dxcol':dict(plot=plot_rlMeans),
                            },
                 # figsize=figsize,
                 **kwargs) as ses:
        ses.plt = plt
 
        
        #plot discretization figures (very slow)
        #ses.plot_xmDisc()
        
        #nice plot per-func of means at different ag levels
        """workaround as this is a combined calc+plotting func"""
        if plot_rlMeans:
            ses.build_rl_dxcol(plot=plot_rlMeans)
 
        
        #combined box plots
        ses.plot_eA_box(grp_colns = ['model_id', 'sector_attribute'])
        
        #per-model bar plots
        ses.plot_eA_bars()
        
        #calc some stats and write to xls
        ses.run_err_stats()
        
        
 
         
        
        

 
        
        out_dir = ses.out_dir
        
    return out_dir

def r1_3mods(#just those used in p2
             #reversed delta values
        ):
    
    return run_aggErr1(
        
            #model selection
            tag='r1_3mods',
            vid_l=[798,811, 49] ,
            
                     
            #run control
            overwrite=True,
            rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 30, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             5, 
                             100,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=2000,  #number of depths to draw from the depths distritupion
                          ),
            
            plot_rlMeans=True,
                 
                 
            compiled_fp_d = {
 
                        },
        
        )
    
def all_r0(#results presented at supervisor meeting on Jan 4th
           #focused on vid 027, 811, and 798
           #but included some stats for every curve in the library
           #the majority of these are FLEMO curves
           #takes ~1hr to run
        ):
    
    return run_aggErr1(
        
            #model selection
            tag='r0',
            #vid_l=[811,798, 410] ,
            selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              3, #flemo 
                              4, 6, 
                              7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
            #run control
            overwrite=False,
            rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 30, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             5, 
                             100,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=2000,  #number of depths to draw from the depths distritupion
                          ),
            
            plot_rlMeans=True,
                 
                 
            compiled_fp_d = {
                'rl_xmDisc_dxcol':  r'C:\LS\10_OUT\2112_Agg\outs\pdist\r0\20220205\working\aggErr1_r2_0104_rl_xmDisc_dxcol.pickle',
                'rl_xmDisc_xvals':  r'C:\LS\10_OUT\2112_Agg\outs\pdist\r0\20220205\working\aggErr1_r2_0104_rl_xmDisc_xvals.pickle',
                'rl_dxcol':         r'C:\LS\10_OUT\2112_Agg\outs\pdist\r0\20220205\working\aggErr1_r3_0104_rl_dxcol.pickle',
                'model_metrics':    r'C:\LS\10_OUT\2112_Agg\outs\pdist\r0\20220205\working\aggErr1_r3_0104_model_metrics.pickle'
                        },
        
        )
    
def r0_noFlemo(
        
        ):
    return run_aggErr1(
            #model selection
            tag='r0_noFlemo',
            #vid_l=[811,798, 410] ,
            selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              #3, #flemo 
                              4, 6, 
                              7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
            #run control
            overwrite=False,
            rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 30, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             5, 
                             100,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=2000,  #number of depths to draw from the depths distritupion
                          ),
            
            plot_rlMeans=True)

def dev(
        
        ):
    
    return run_aggErr1(
        
            #model selection
            tag='dev',
            #=======================================================================
            # vid_l=[
            #         796, #Budiyono (2015) 
            #        #402, #MURL linear
            #        #852, #Dutta (2003) nice and parabolic
            #        #33, #FLEMO resi...
            #        #332, #FLEMO commericial
            #        ], #running on a single function
            #=======================================================================
        
            #vid_l=[811,798, 410] ,
            vid_sample = 3,
            selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              3, #flemo 
                              4, 6, 
                              7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
            #run control
            overwrite=True,
            rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 5, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             #5, 
                             100,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=100,  #number of depths to draw from the depths distritupion
                          ),
            
            plot_rlMeans=True,
                 
                 
            compiled_fp_d = {
                    #===========================================================
                    # 'rl_xmDisc_dxcol':r'C:\LS\10_OUT\2112_Agg\outs\pdist\dev\20220205\pdist_dev_0205_rl_xmDisc_dxcol.pickle',
                    # 'rl_xmDisc_xvals':r'C:\LS\10_OUT\2112_Agg\outs\pdist\dev\20220205\pdist_dev_0205_rl_xmDisc_xvals.pickle',
                    # 'rl_dxcol':r'C:\LS\10_OUT\2112_Agg\outs\pdist\dev\20220205\pdist_dev_0205_rl_dxcol.pickle',
                    # 'model_metrics':r'C:\LS\10_OUT\2112_Agg\outs\pdist\dev\20220205\pdist_dev_0205_model_metrics.pickle',
                    #===========================================================
                        },
        
        )

if __name__ == "__main__": 
    
    output=r1_3mods()
    
    
 
    #output=all_r0()
    #output=dev()
    #output = run_plotVfunc()
 
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))