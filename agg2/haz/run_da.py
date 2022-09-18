'''
Created on Sep. 9, 2022

@author: cefect
'''
import os, pathlib, math
from definitions import proj_lib
from hp.basic import get_dict_str
import pandas as pd
idx = pd.IndexSlice

def SJ_plots_0910(        
        fp_lib = {
            'filter':{
                's2':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r6\SJ\filter\20220910\stats\SJ_r6_filter_0910_stats.pkl',
                's1':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r6\SJ\filter\20220910\statsF\SJ_r6_filter_0910_statsF.pkl',
                #'diff':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\filter\\20220909\\errStats\\SJ_r6_filter_0909_errStats.pkl',                
                },
            'direct':{
                's2': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r6\SJ\direct\20220910\stats\SJ_r6_direct_0910_stats.pkl',
                's1': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r6\SJ\direct\20220910\statsF\SJ_r6_direct_0910_statsF.pkl',
                #'diff': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\errStats\\SJ_r6_direct_0909_errStats.pkl',                
                
                }
            }
        ):
    return run_haz_plots(fp_lib, proj_name='SJ', run_name='r4_da')

def SJ_plots_0909(        
        fp_lib = {
            'filter':{
                's2':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\filter\\20220909\\stats\\SJ_r6_filter_0909_stats.pkl',
                's1':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\filter\\20220909\\statsF\\SJ_r6_filter_0909_statsF.pkl',
                'diff':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\filter\\20220909\\errStats\\SJ_r6_filter_0909_errStats.pkl',                
                },
            'direct':{
                's2': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\stats\\SJ_r6_direct_0909_stats.pkl',
                's1': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\statsF\\SJ_r6_direct_0909_statsF.pkl',
                'diff': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\errStats\\SJ_r6_direct_0909_errStats.pkl',                
                
                }
            }
        ):
    return run_haz_plots(fp_lib, proj_name='SJ', run_name='r4_da')


def SJ_plots_0830(
        fp_d  = {
            'direct':r'C:\LS\10_IO\2112_Agg\outs\SJ\r4_direct\20220908\stats\SJ_r4_direct_0908_stats.pkl',
            'directF':r'C:\LS\10_OUT\2112_Agg\outs\SJ\r4_direct\20220908\statsF\SJ_r4_direct_0908_statsF.pkl',
            'filter':r'C:\LS\10_IO\2112_Agg\outs\SJ\r4_filter\20220908\stats\SJ_r4_filter_0908_stats.pkl', 
            'filterF':r'C:\LS\10_OUT\2112_Agg\outs\SJ\r4_filter\20220908\statsF\SJ_r4_filter_0908_statsF.pkl'           
            }):
    return run_haz_plots(fp_d)

def run_haz_plots(fp_lib,
                  **kwargs):
    """construct figure from SJ downscale cat results"""
    from agg2.haz.da import UpsampleDASession as Session
    from hp.pd import view
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
 
    
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(fp_lib['filter']['s2'])).parents[3], #C:/LS/10_OUT/2112_Agg/outs/agg2/r5
        'da', today_str)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, **kwargs) as ses:
        
        #=======================================================================
        # data prep
        #=======================================================================
        #join the simulation results (and clean up indicides
        dxcol_raw = ses.join_stats(fp_lib, write=False)
        
 
        #add residuals  
        dxcol1 = dxcol_raw.join(pd.concat([dxcol_raw['s2']-dxcol_raw['s1']], names=['base'], keys=['s12'], axis=1)).copy()
        
        print({lvlName:i for i, lvlName in enumerate(dxcol1.columns.names)})
        
        #=======================================================================
        # #add residuals normalized        
        #=======================================================================
        """probably some way to do this natively w/ panda (transform?)
        but couldnt figure out how to divide across 2 levels
        """
        base_dxcol = dxcol1.loc[1, idx['s1', 'direct',:,'full', :]].droplevel((0,1,3), axis=1).reset_index(drop=True) #baseline values
        
        d = dict()
        for layName, gdx in dxcol1['s12'].groupby('layer', axis=1):
            base_ser = base_dxcol[layName].iloc[0,:]
            d[layName] = gdx.droplevel('layer',axis=1).divide(base_ser, axis=1, level='metric')
        div_dxcol = pd.concat(d, axis=1, names=['layer'])
 
        
        dxcol1 = dxcol1.join(
            pd.concat([div_dxcol], names=['base'], keys=['s12N'], axis=1).reorder_levels(dxcol1.columns.names, axis=1)
            )
 
 
        print(dxcol1.columns.get_level_values('metric').unique().tolist())
        metrics_l = ['mean', 'posi_area', 'vol']
        
 
        
 
        
        #=======================================================================
        # lines on residuals (s12)
        #=======================================================================
 
        #=======================================================================
        # dxcol3 = dxcol1.loc[:, idx['s12', :, :, coln_l]].droplevel(0, axis=1)
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
 #        dxcol3 = dxcol1.loc[:, idx['s12N', :, 'wd',:, metrics_l]].droplevel(['base', 'layer'], axis=1)
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
        # with wse RMSE
        #=======================================================================
        #join the differences
        #=======================================================================
        # metric2 = 'meanErr'
        # lab_d = {
        #     'meanErr':r'$\overline{WSE_{s2,i} - WSE_{s1,i}}$',
        #     #'meanErr':r'$\frac{\sum_{i}^{N_{12}} WSE_{s2,i} - WSE_{s1,i}}{N_{12}}$',
        #     }
        # 
        # 
        # 
        # dxcol4 = dxcol3.join(dxcol1.loc[:, idx['diff', :, 'wse',:, metric2]].droplevel(['base', 'layer'], axis=1)) 
        # 
        # serx = dxcol4.stack(level=dxcol4.columns.names).sort_index(sort_remaining=True
        #                                ).reindex(index=metrics_l+[metric2], level='metric'
        #                                 ).droplevel(['scale', 'pixelArea'])
        #                                 
        #                                 
        # ses.plot_matrix_metric_method_var(serx,
        #                                   map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'pixelLength'},
        #                                   ylab_d={
        #                                       'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$', 
        #                                       'mean':r'$\frac{\overline{WD_{s2}}-\overline{WD_{s1}}}{\overline{WD_{s1}}}$', 
        #                                       'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
        #                                       metric2:lab_d[metric2]},
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd_%s.svg'%metric2),
        #                                   matrix_kwargs = dict(figsize=(6.5,7.25))
        #                                   )
        #=======================================================================
        
        #=======================================================================
        # with WSE mean
        #=======================================================================
        
   #============================================================================
   #      dxcol4 = dxcol1.loc[:, idx['s12N', :, ('wd', 'wse'),:, metrics_l]].droplevel('base', axis=1)
   #        
   #      #stack into a series
   #      serx1 = dxcol4.stack(level=dxcol4.columns.names).sort_index(sort_remaining=True
   #                                     ).reindex(index=metrics_l, level='metric'
   #                                      ).droplevel(['scale', 'pixelArea'])
   #                                        
   #      #concat layer and metric
   #      """because for plotting we treat this as combined as 1 dimension"""
   #      mcoln = 'layer_metric'
   #      df = serx1.index.to_frame().reset_index(drop=True)
   #      df[mcoln] = df['layer'].str.cat(df['metric'], sep='_')
   # 
   #      serx = pd.Series(serx1.values, index = pd.MultiIndex.from_frame(df.drop(['layer', 'metric'], axis=1)))
   #        
   #      #sort
   #      m1_l = ['wd_mean', 'wse_mean', 'wd_posi_area', 'wd_vol']
   #      serx = serx.reindex(index=m1_l, level=mcoln) #apply order
   #        
   #        
   #      #plot
   #      ses.plot_matrix_metric_method_var(serx,
   #                                        map_d = {'row':mcoln,'col':'method', 'color':'dsc', 'x':'pixelLength'},
   #                                        ylab_d={
   #                                            'wd_vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$', 
   #                                            'wd_mean':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$', 
   #                                            'wd_posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
   #                                            'wse_mean':r'$\frac{\overline{WSE_{s2}}-\overline{WSE_{s1}}}{\overline{WSE_{s1}}}$', 
   #                                            },
   #                                        ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd_wse.svg'),
   #                                        matrix_kwargs = dict(figsize=(6.5,7.25)),
   #                                        ax_lims_d = {
   #                                            'y':{'wd_mean':(-1.5, 0.2), 'wse_mean':(-0.1, 1.5), 'wd_posi_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1)},
   #                                            }
   #                                        )
   #============================================================================
        
        #=======================================================================
        # for presentation (WD and A)
        #=======================================================================
        m_l = ['mean', 'posi_area']
        dx1 = dxcol1.loc[:, idx['s12N', :, 'wd',:, m_l]
                         ].droplevel(['base', 'layer'], axis=1).droplevel(('scale', 'pixelArea')
                              ).drop('direct', level='method',axis=1)
          
        #stack into a series
        serx = dx1.stack(level=dx1.columns.names).sort_index(sort_remaining=True
                                       ).reindex(index=m_l, level='metric') 
          
          
        #plot
        ses.plot_matrix_metric_method_var(serx,
                                          map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'pixelLength'},
                                          ylab_d={
                                            #'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$', 
                                            'mean':r'$\frac{\overline{WD_{s2}}-\overline{WD_{s1}}}{\overline{WD_{s1}}}$', 
                                            'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
                                              },
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd_present.svg'),
                                          matrix_kwargs = dict(figsize=(3,4), set_ax_title=False, add_subfigLabel=False),
                                          ax_lims_d = {
                                              'y':{'mean':(-1.5, 0.2),  'posi_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1)},
                                              },
                                          output_fig_kwargs=dict(transparent=False, add_stamp=False),
                                          legend_kwargs=dict(loc=3),
                                          ax_title_d=dict(), #no axis titles
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
    SJ_plots_0910()
        
