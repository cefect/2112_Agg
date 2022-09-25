'''
Created on Sep. 9, 2022

@author: cefect
'''
import faulthandler
faulthandler.enable()

import os, pathlib, math, pprint
from definitions import proj_lib
from hp.basic import get_dict_str, today_str
import pandas as pd
idx = pd.IndexSlice


from agg2.haz.run import res_fp_lib as haz_res_fp_lib


res_fp_lib = {
    'r9':{
            'filter':{  
                's2': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\stats\\SJ_r9_filter_0921_stats.pkl',
                's1': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\statsF\\SJ_r9_filter_0921_statsF.pkl',
                'diffs': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220922\\diffStats\\SJ_r9_filter_0922_diffStats.pkl',
                },
            'direct':{  
                's2': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\stats\\SJ_r9_direct_0921_stats.pkl',
                's1': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\statsF\\SJ_r9_direct_0921_statsF.pkl',
                'diffs': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220922\\diffStats\\SJ_r9_direct_0922_diffStats.pkl'
                }
            },
    'r10t':{
        'direct':r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\direct\20220925\hstats\20220925\SJ_r1_hs_0925_stats.pkl',
        'filter':r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\filter\20220925\hstats\20220925\SJ_r1_hs_0925_stats.pkl',
        },
    'r10':{
        'direct':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\direct\20220925\hstats\20220925\SJ_r10_hs_0925_stats.pkl',
        'filter':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\filter\20220925\hstats\20220925\SJ_r10_hs_0925_stats.pkl'
        }
    }

 

def SJ_da_run(
        run_name='r9'
        ):    
    
    return run_haz_plots(res_fp_lib[run_name], proj_name='SJ', run_name=run_name)

def SJ_dev_run(
        fp_d = {
            'filter':r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\filter\hstats\20220925\SJ_r1_hs_0925_stats.pkl',
            'direct':r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\direct\hstats\20220925\SJ_r1_hs_0925_stats.pkl'
            
            }
        
        ):
    return run_haz_plots(fp_d, proj_name='SJ', run_name='t')

 



def log_dxcol(log, dxcol_raw):
    mdex = dxcol_raw.columns
    log.info(
        f'for {str(dxcol_raw.shape)}' + 
        '\n    base:     %s' % mdex.unique('base').tolist() + 
        '\n    layers:   %s' % mdex.unique('layer').tolist() + 
        '\n    metrics:  %s' % mdex.unique('metric').tolist())
    return 




def run_haz_plots(fp_lib,
                  write=True,
                  **kwargs):
    """construct figure from SJ downscale cat results"""
    from agg2.haz.da import UpsampleDASession as Session
    from hp.pd import view
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(fp_lib['filter'])).parents[1],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
        'da', 'haz', today_str)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, **kwargs) as ses:
        """for haz, working with aggregated zonal stats.
            these are computed on:
                aggregated (s2) data with UpsampleSession.run_stats()
                raw/fine (s1) data with UpsampleSession.run_stats_fine()
                local diff (s2-s1) with UpsampleSession.run_diff_stats()
            
 
        """
        idxn = ses.idxn
        log = ses.logger
        #=======================================================================
        # DATA PREP---------
        #=======================================================================
 
        # join the simulation results (and clean up indicides
        d = {k:pd.read_pickle(fp) for k,fp in fp_lib.items()}
        dxcol_raw = pd.concat(d, axis=1, names=['method']).swaplevel('base', 'method', axis=1).sort_index(axis=1)
        #dxcol_raw = ses.join_stats(fp_lib, write=False)
        del d
        log_dxcol(log, dxcol_raw)
                 
        
        """
        view(dxcol_raw.T)
        """
        
 
 

        
        """Windows fatal exception: access violation
        but only when I assign this to a variable used below?
        """  
        dx2 = ses.data_prep(dxcol_raw) 
 
        """
        view(dx2.T)
        view(dx2['diffs'].T)
        """
        #dx2.index = pd.MultiIndex.from_frame(scale_df.reset_index())
 
        


        
        #=======================================================================
        # HELPERS------
        #=======================================================================
        def cat_mdex(mdex, levels=['layer', 'metric']):
            """concatnate two levels into one of an mdex"""
            mcoln = '_'.join(levels)
            df = mdex.to_frame().reset_index(drop=True)
 
            df[mcoln] = df[levels[0]].str.cat(df[levels[1]], sep='_')
            
            return pd.MultiIndex.from_frame(df.drop(levels, axis=1)), mcoln
        
        #=======================================================================
        # AGG ZONAL PLOTS---------
        #=======================================================================
        #=======================================================================
        # lines on residuals (s12)
        #=======================================================================
 
        #=======================================================================
        # dxcol3 = dx1.loc[:, idx['s12', :, :, coln_l]].droplevel(0, axis=1)
        # serx = dxcol3.unstack().reindex(index=coln_l, level=2) 
        #=======================================================================
        
        #=======================================================================
        # ses.plot_matrix_metric_method_var(serx,
        #                                   map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'downscale'},
        #                                   ylab_d={
        #                                       'vol':'$\sum V_{s2}-\sum V_{s1}$ (m3)', 
        #                                       'wd_mean':'$\overline{WD_{s2}}-\overline{WD_{s1}}$ (m)', 
        #                                       'wse_area':'$\sum A_{s2}-\sum A_{s1}$ (m2)'},
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_var_resid.svg'))
        #=======================================================================
        
        #=======================================================================
        # lines on residuals NORMALIZED (s12N)
        #=======================================================================
 #==============================================================================
 #        #just water depth and the metrics
 #        dxcol3 = dx1.loc[:, idx['s12N', :, 'wd',:, metrics_l]].droplevel(['base', 'layer'], axis=1)
 # 
 #        #stack into a series
 #        serx = dxcol3.stack(level=dxcol3.columns.names).sort_index(sort_remaining=True
 #                                       ).reindex(index=metrics_l, level='metric'
 #                                        ).droplevel(['scale', 'pixelArea'])
 #==============================================================================
 
        #=======================================================================
        # ses.plot_matrix_metric_method_var(serx,
        #                                   map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'pixelLength'},
        #                                   ylab_d={
        #                                       'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$', 
        #                                       'mean':r'$\frac{\overline{WD_{s2}}-\overline{WD_{s1}}}{\overline{WD_{s1}}}$', 
        #                                       'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$'},
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd.svg'))
        #=======================================================================
        
 
        
        #=======================================================================
        # four metrics
        #=======================================================================

        
        mcoln = 'layer_metric'
        m1_l = ['wd_mean', 'wse_mean', 'wse_real_area', 'wd_vol']

#===============================================================================
#         def get_stack(baseName, 
#                       metrics_l=['mean', 'posi_area', 'vol'],
#                       metrics_cat_l=None,
#                       ):
#             """common collection of data stack
#              
#              Parameters
#             -----------
#             metrics_cat_l: list
#                 post concat order
#             """
#             
#             dxi = dx2[baseName]
#             #check all the metrics are there
#             assert set(metrics_l).difference(dxi.columns.unique('metric'))==set(), f'requested metrics no present on {baseName}'
#             
#             dxi1 = dxi.loc[:, idx[:, ('wd', 'wse'),:, metrics_l]]
#             
#             assert not dxi1.isna().all().all()
#             
#             
#             dxi1.columns, mcoln = cat_mdex(dxi1.columns) #cat layer and metric
#             
#             
#             if not metrics_cat_l is None:
#                 assert set(dxi1.columns.unique('layer_metric')).symmetric_difference(m1_l)==set(), 'metric list mismatch'
#             else:
#                 metrics_cat_l = dxi1.columns.unique('layer_metric').tolist()
#                
#             # stack into a series
#             serx = dxi1.stack(level=dxi1.columns.names).sort_index(sort_remaining=True
#                                            ).reindex(index=metrics_cat_l, level=mcoln
#                                             ).droplevel(['scale', 'pixelArea'])
#             assert len(serx)>0
#  
#                                             
# 
#             return serx, mcoln
#===============================================================================
 
        
        #=======================================================================
        # single-base raster stats
        #=======================================================================
        """
        s1 methods:
            these should be the same (masks are the same, baseline is the same)
             
        """
        #=======================================================================
        # for baseName in [
        #     's2', 's1', 's12', 
        #     #'diffs','diffsN', #these are granular
        #     ]:
        #     log.info(f'plotting {baseName} \n\n')
        #     serx = get_stack(baseName)
        #         
        #     # plot
        #     ses.plot_matrix_metric_method_var(serx,
        #                                       map_d={'row':mcoln, 'col':'method', 'color':'dsc', 'x':'pixelLength'},
        #                                       ylab_d={},
        #                                       ofp=os.path.join(ses.out_dir, 'metric_method_var_%s.svg' % baseName),
        #                                       matrix_kwargs=dict(figsize=(6.5, 7.25), set_ax_title=False, add_subfigLabel=True),
        #                                       ax_lims_d={
        #                                           # 'y':{'wd_mean':(-1.5, 0.2), 'wse_mean':(-0.1, 1.5), 'wse_real_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1)},
        #                                           }
        #                                       )
        #=======================================================================
        
        #=======================================================================
        # resid normed.  Figure 5: Bias from upscaling 
        #=======================================================================

        """
        direct:
            why is direct flat when s2 is changing so much?
                because s1 and s2 are identical
                remember... direct just takes the zonal average anyway
                so the compute metric is the same as the stat
                
        wse:
            dont want to normalize this one
            
         
        no area bias?
        direct: delta WSE has changed slightly
        volume bias is normed strange
        
        """
        
        #dx2.reset_index(drop=True).loc[(0,3,6), idx[:, 'direct', ']]

        #=======================================================================
        # baseName = 's12N'
        # serx = get_stack(baseName) 
        #=======================================================================
        """join wd with wse from different base"""
        dxi = pd.concat([
            dx2.loc[:, idx['s12AN',:, 'wd',:, ('mean', 'vol')]],
            dx2.loc[:, idx['s12AN',:, 'wse',:, 'real_area']],
            dx2.loc[:, idx['s12',:, 'wse',:, 'mean']],
            #dx2.loc[:, idx['s12AN',:, 'wse',:, 'real_area']]
            ], axis=1).droplevel('base', axis=1).sort_index(axis=1)
                                            
        dxi.columns, mcoln = cat_mdex(dxi.columns) #cat layer and metric
          
        serx = dxi.droplevel((0,2), axis=0).stack(dxi.columns.names
                          ).reindex(index=['full', 'DD', 'DP', 'WP', 'WW'], level='dsc' #need a bucket with all the metrics to be first
                        ).reindex(index=m1_l, level=mcoln) 
        """
        view(serx)
        """
             
        # plot
        ses.plot_matrix_metric_method_var(serx,
                                          map_d={'row':mcoln, 'col':'method', 'color':'dsc', 'x':'pixelLength'},
                                          ylab_d={
                                              'wd_vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$',
                                              'wd_mean':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$',
                                              'wse_real_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
                                              'wse_mean':r'$\overline{WSE_{s2}}-\overline{WSE_{s1}}$',
                                              },
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_%s.svg' % ('s12_s12N')),
                                          matrix_kwargs=dict(figsize=(6.5, 7.25), set_ax_title=False, add_subfigLabel=True),
                                          ax_lims_d={
                                              'y':{'wd_mean':(-1.5, 0.2), 'wse_real_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1),
                                                   'wse_mean':(-1.0, 15.0),
                                                   },
                                              }
                                          )
        
 

        return 
        #=======================================================================
        # for presentation (WD and A)
        #=======================================================================
        #=======================================================================
        # m_l = ['mean', 'posi_area']
        # dx1 = dx1.loc[:, idx['s12N', :, 'wd',:, m_l]
        #                  ].droplevel(['base', 'layer'], axis=1).droplevel(('scale', 'pixelArea')
        #                       ).drop('direct', level='method',axis=1)
        #   
        # #stack into a series
        # serx = dx1.stack(level=dx1.columns.names).sort_index(sort_remaining=True
        #                                ).reindex(index=m_l, level='metric') 
        #   
        #   
        # #plot
        # ses.plot_matrix_metric_method_var(serx,
        #                                   map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'pixelLength'},
        #                                   ylab_d={
        #                                     #'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$', 
        #                                     'mean':r'$\frac{\overline{WD_{s2}}-\overline{WD_{s1}}}{\overline{WD_{s1}}}$', 
        #                                     'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
        #                                       },
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd_present.svg'),
        #                                   matrix_kwargs = dict(figsize=(3,4), set_ax_title=False, add_subfigLabel=False),
        #                                   ax_lims_d = {
        #                                       'y':{'mean':(-1.5, 0.2),  'posi_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1)},
        #                                       },
        #                                   output_fig_kwargs=dict(transparent=False, add_stamp=False),
        #                                   legend_kwargs=dict(loc=3),
        #                                   ax_title_d=dict(), #no axis titles
        #                                   )
        #=======================================================================
        #=======================================================================
        # GRANULAR ZONAL---------
        #=======================================================================
        #=======================================================================
        # single-base raster stats
        #=======================================================================
        """
        diffs
            direct
                wd_meanErr:
                    why so different from Agg?
                    
                    the routine (and the agg stat) are
                    
        need to plot populations... 
 
             
        """
        #dx2['diffs'].columns.unique('metric').to_list()
        for baseName in [
            'diffs',
           'diffsN', #these are granular
            ]:
            log.info(f'\n\nplotting {baseName} \n\n')
            serx, mcoln = get_stack(baseName, metrics_l=['meanErr', 'meanAbsErr'])
                
            # plot
            ses.plot_matrix_metric_method_var(serx,title=baseName,
                                              map_d={'row':mcoln, 'col':'method', 'color':'dsc', 'x':'pixelLength'},
                                              ylab_d={},
                                              ofp=os.path.join(ses.out_dir, 'metric_method_var_%s.svg' % baseName),
                                              matrix_kwargs=dict(figsize=(6.5, 7.25), set_ax_title=False, add_subfigLabel=True),
                                              ax_lims_d={
                                                  # 'y':{'wd_mean':(-1.5, 0.2), 'wse_mean':(-0.1, 1.5), 'wse_real_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1)},
                                                  }
                                              )
        
        #=======================================================================
        # stackced areas ratios
        #=======================================================================
        
        #=======================================================================
        # #compute fraction
        #=======================================================================
        
        #=======================================================================
        # #reduce
        # dx1 = dxcol_raw.loc[:, idx['s2',:,'wd',:,'post_count']].droplevel(('base', 'metric', 'layer'), axis=1).droplevel((1,2), axis=0)
        # df1 = dx1.drop('full',level='dsc', axis=1).dropna().drop('filter',level='method', axis=1).droplevel('method', axis=1)
        #  
        # #compute fraction
        # fdf = df1.divide(df1.sum(axis=1), axis=0)
        #   
        # #plot
        # ses.plot_dsc_ratios(fdf)
        #=======================================================================
        
        
if __name__ == "__main__":

  
    SJ_da_run(run_name='r10')
        
