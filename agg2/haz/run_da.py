'''
Created on Sep. 9, 2022

@author: cefect
'''
import os, pathlib
from definitions import proj_lib
from hp.basic import get_dict_str
import pandas as pd
idx = pd.IndexSlice


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
    out_dir = pathlib.Path(os.path.dirname(fp_lib['filter']['s2'])).parents[3] #C:/LS/10_OUT/2112_Agg/outs/agg2/r5
    
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
 
        
 
        
        """
        dxcol_raw.columns.levels
        
        open_pick(fp_d['directF'])
        view(dxcol_raw)
        dxcol_raw.columns
        """
        print(dxcol1.columns.get_level_values('metric').unique().tolist())
        metrics_l = ['mean', 'posi_area', 'vol']
        
 
        
        #=======================================================================
        # lines on s2: row:metric, col:method, color:dsc
        #=============================================R==========================
 #==============================================================================
 #        dxcol2 = dxcol1.copy()
 #        #promote pixelLength to index
 #        map_ser = dxcol1.loc[:, idx['s2','direct','full','pixelLength']].rename('pixelLength').astype(int)        
 #        dxcol2.index = pd.MultiIndex.from_frame(dxcol2.index.to_frame().join(map_ser))
 #        
 #        
 #        serx = dxcol2.loc[:, idx['s2', :, :, coln_l]].droplevel(0).droplevel(0, axis=1).unstack().reindex(index=coln_l, level=2) 
 #        
 # 
 # 
 #        #ses.plot_matrix_metric_method_var(serx)
 #==============================================================================
        
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
        #just water depth and the metrics
        dxcol3 = dxcol1.loc[:, idx['s12N', :, 'wd',:, metrics_l]].droplevel(['base', 'layer'], axis=1)
 
        #stack into a series
        serx = dxcol3.stack(level=dxcol3.columns.names).sort_index(sort_remaining=True
                                       ).reindex(index=metrics_l, level='metric'
                                        ).droplevel(['scale', 'pixelArea'])
                                        
        """
        view(serx)
        """
 
        

        
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
        metric2 = 'meanErr'
        dxcol4 = dxcol3.join(dxcol1.loc[:, idx['diff', :, 'wse',:, metric2]].droplevel(['base', 'layer'], axis=1)) 
        
        serx = dxcol4.stack(level=dxcol4.columns.names).sort_index(sort_remaining=True
                                       ).reindex(index=metrics_l+[metric2], level='metric'
                                        ).droplevel(['scale', 'pixelArea'])
                                        
                                        
        ses.plot_matrix_metric_method_var(serx,
                                          map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'pixelLength'},
                                          ylab_d={
                                              'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$', 
                                              'mean':r'$\frac{\overline{WD_{s2}}-\overline{WD_{s1}}}{\overline{WD_{s1}}}$', 
                                              'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
                                              metric2:'WSE %s (m)'%metric2},
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd_%s.svg'%metric2),
                                          matrix_kwargs = dict(figsize=(6.5,8))
                                          )
        
        #=======================================================================
        # stackced areas ratios
        #=======================================================================
        
        #=======================================================================
        # #compute fraction
        #=======================================================================
        #=======================================================================
        # coln = 'frac'
        # dxcol4 = dxcol2.loc[:, idx['s2',:,:,:]].droplevel(0).droplevel(0, axis=1)
        # #view(dxcol.loc[:, idx[:, :, 'count']])
        # #get just the vount values
        # cnt_dx = dxcol4.loc[:, idx[:, :, 'count']]
        # 
        # #divide by the total        
        # cnt_dx = cnt_dx.divide(cnt_dx.loc[:, idx[:, 'all', :]].droplevel([1,2], axis=1), axis=0, level=0).droplevel(-1, axis=1)
        # 
        # #add the new label
        # cnt_dx = pd.concat([cnt_dx], keys=[coln], names=['metric'], axis=1).reorder_levels(dxcol4.columns.names, axis=1)
        # 
        # dxcol4 = dxcol4.join(cnt_dx).sort_index(axis=1)
        #  
        # #slice to just these
        # dfi = dxcol4.loc[:, idx['direct', :, 'frac']].droplevel([0,2], axis=1).drop('all', axis=1)
        # 
        # #plot
        # ses.plot_dsc_ratios(dfi.dropna())
        #=======================================================================
        
        
        
if __name__ == "__main__":
    SJ_plots_0909()
        