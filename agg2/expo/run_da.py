'''
Created on Sep. 10, 2022

@author: cefect
'''
import os, pathlib
from definitions import proj_lib
from hp.basic import get_dict_str
import pandas as pd
idx = pd.IndexSlice

def SJ_plots_0910(
        fp_lib = {
                'direct':{
                    'catMasks': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r7\\SJ\\direct\\20220910\\cMasks\\SJ_r7_direct_0910_cMasks.pkl',
                    'arsc':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\direct\20220917\arsc\SJ_r7_direct_0917_arsc.pkl',
                    'wd':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\direct\20220911\lsamp_wd\SJ_r7_direct_0911_lsamp_wd.pkl',
                    'wse':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\direct\20220911\lsamp_wse\SJ_r7_direct_0911_lsamp_wse.pkl',
                    },
                'filter':{
                    'catMasks':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
                    #'arsc':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r1\SJ\filter\20220910\arsc\SJ_r1_filter_0910_arsc.pkl',
                    'wd':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wd\SJ_r7_filter_0911_lsamp_wd.pkl',
                    'wse':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wse\SJ_r7_filter_0911_lsamp_wse.pkl'
                    }
                },
        ):
    return run_plots(fp_lib)

def run_plots(fp_lib,
                  **kwargs):
    """construct figure from SJ downscale cat results"""
    from agg2.expo.da import ExpoDASession as Session
    from hp.pd import view
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    out_dir = pathlib.Path(os.path.dirname(fp_lib['filter']['catMasks'])).parents[3] #C:/LS/10_OUT/2112_Agg/outs/agg2/r5
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, **kwargs) as ses:
        
        #=======================================================================
        # data prep
        #=======================================================================

        
        #get the rsc for each asset and scale        
        dsc_df = pd.read_pickle(fp_lib['direct']['arsc'])
        
        #join the simulation results (and clean up indicides
        raw_dx = ses.join_layer_samps(fp_lib, write=False, dsc_df=dsc_df)
        
 
        #=======================================================================
        # agg stats----------
        #=======================================================================
        sdx = ses.get_dsc_stats(raw_dx)   #compute zonal stats
        
        """
        view(sdx1.T)
        """
        
                        
        #compute residual
        sdx1 = pd.concat({'samps':sdx, 'resid':sdx.subtract(sdx.loc['full', :])}, axis=1, names=['base'])
        
        #residuals normalized
        ndx = adx1.loc[:, 'resid'].divide(adx1.loc[1, 'samps'])
        adx2 = adx1.join(pd.concat({'residN':ndx}, names=['base'], axis=1))
        
        """
        view(adx2.loc[:, idx[:, 'direct', 'wd', 'mean']])
        """
        
        #=======================================================================
        # plot resid normd
        #=======================================================================
        #just water depth and the metrics
        dx = pd.concat({'full':adx2['residN'].loc[:, idx[:, :, 'mean']].droplevel('metric', axis=1)}, names=['dsc'], axis=1)
 
  
        #stack into a series
        serx = dx.stack(level=dx.columns.names)
 
 
        ses.plot_matrix_metric_method_var(serx,
                                          map_d = {'row':'layer','col':'method', 'color':'dsc', 'x':'scale'},
                                          ylab_d={
                                            'wd':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$', 
                                            'wse':r'$\frac{\overline{WSE_{s2}}-\overline{WSE_{s1}}}{\overline{WSE_{s1}}}$', 
                                            #'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
                                              },
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd_assets.svg'))
        
        #=======================================================================
        # item residuals
        #=======================================================================
        """NOTE: for WSE this returns nulls if any value is null"""
        ldx1 = lsamp_dx.subtract(lsamp_dx.loc[:, idx[:, :, 1]].droplevel('scale', axis=1), axis=1) 
        
        ldx2 = pd.concat({'samps':lsamp_dx, 'resid':ldx1}, axis=1, names=['base'])
 
        
        #=======================================================================
        # stackced areas ratios
        #=======================================================================
 #==============================================================================
 #        #compute counts
 #        dx1 = arsc_dx.sum(axis=0).astype(int).rename('count').to_frame()
 #        
 #        #compute fraction        
 #        tdx = dx1.groupby(level=('method', 'scale'), axis=0).sum()  #totals per scale (roughly equal)         
 #              
 #        fdx = dx1.divide(tdx).iloc[:,0].rename('frac').sort_index() #divide by the total
 #        """two methods should hvae identical cat masks"""
 #        
 #        fdf = fdx.drop('filter').droplevel(0).unstack() #reduce
 # 
 #         
 #        #plot
 #        ses.plot_dsc_ratios(fdf, ylabel='asset pop. fraction')
 #==============================================================================
        """
        view(fdf)
        """
        
        
        
        
if __name__ == "__main__":
    SJ_plots_0910()