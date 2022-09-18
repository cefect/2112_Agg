'''
Created on Sep. 10, 2022

@author: cefect

exposure data analysis
'''
import os, pathlib
from definitions import proj_lib
from hp.basic import get_dict_str, today_str
from hp.pd import append_levels, view
import pandas as pd
idx = pd.IndexSlice

def SJ_plots_0910(
        fp_lib = {
                'direct':{
                    'catMasks': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\SJ\direct\20220917\cMasks\SJ_r8_direct_0917_cMasks.pkl',
                    'arsc':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\SJ\direct\20220918\arsc\SJ_r8_direct_0918_arsc.pkl',
                    'wd':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\SJ\direct\20220918\lsamp_wd\SJ_r8_direct_0918_lsamp_wd.pkl',
                    'wse':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\SJ\direct\20220918\lsamp_wse\SJ_r8_direct_0918_lsamp_wse.pkl',
                    },
                'filter':{
                    'catMasks':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
                    #'arsc':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r1\SJ\filter\20220910\arsc\SJ_r1_filter_0910_arsc.pkl',
                    'wd':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wd\SJ_r7_filter_0911_lsamp_wd.pkl',
                    'wse':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wse\SJ_r7_filter_0911_lsamp_wse.pkl'
                    }
                },
        run_name='r8'):
    return run_plots(fp_lib)

def run_plots(fp_lib,
                  **kwargs):
    """construct figure from SJ downscale cat results"""
    from agg2.expo.da import ExpoDASession as Session
 
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(fp_lib['filter']['catMasks'])).parents[3], #C:/LS/10_OUT/2112_Agg/outs/agg2/r5
        'da', today_str)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, **kwargs) as ses:
        log = ses.logger
        #=======================================================================
        # data prep
        #=======================================================================

        
        #get the rsc for each asset and scale        
        dsc_df = pd.read_pickle(fp_lib['direct']['arsc'])
        
        #join the simulation results (and clean up indicides
        samp_dx = ses.join_layer_samps(fp_lib)
        
        samp_dx.loc[:, idx['filter', 'wse', (1, 8)]].hist()
        #=======================================================================
        # agg stats----------
        #=======================================================================
        #=======================================================================
        # #compute s2 zonal
        #=======================================================================
        gcols = ['method', 'layer']
        d = dict()
        for i, (gkeys, gdx) in enumerate(samp_dx.groupby(level=gcols, axis=1)):
            gdx_c = pd.concat({gkeys[1]:gdx.droplevel(gcols, axis=1), 'dsc':dsc_df}, axis=1, names=['layer']).sort_index(sort_remaining=True, axis=1)
            dxi = ses.get_dsc_stats1(gdx_c)   #compute zonal stats
            dxi.columns = append_levels(dxi.columns, dict(zip(gcols, gkeys))) #add the levels back
            d[i] = dxi
         
        #merge
        sdx_s2 = pd.concat(d.values(), axis=1).reorder_levels(list(samp_dx.columns.names)+['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
         
        sdx1 = pd.concat({'s2':sdx_s2}, axis=1, names=['base'])
        log.info('built s2 zonal w/ %s'%(str(sdx1.shape)))
        #=======================================================================
        # s1 zonal
        #=======================================================================
        d = dict()
        for i, (gkeys, gdx) in enumerate(samp_dx.groupby(level=gcols, axis=1)):
            #always computing against the fine samples
            gdx0 = pd.concat({res:gdx.droplevel(gcols, axis=1).iloc[:, 0] for res in gdx.columns.unique('scale')}, axis=1, names='scale')
 

            gdx_c = pd.concat({gkeys[1]:gdx0, 'dsc':dsc_df}, axis=1, names=['layer']).sort_index(sort_remaining=True, axis=1)
            dxi = ses.get_dsc_stats1(gdx_c)   #compute zonal stats
            dxi.columns = append_levels(dxi.columns, dict(zip(gcols, gkeys))) #add the levels back
            d[i] = dxi
        
        #merge
        sdx_s1 = pd.concat(d.values(), axis=1).reorder_levels(list(samp_dx.columns.names)+['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
        
        sdx2 = pd.concat([sdx1, pd.concat({'s1':sdx_s1}, axis=1, names=['base'])], axis=1)
        log.info('built s1 zonal w/ %s'%(str(sdx2.shape)))       
        #=======================================================================
        # #compute residual
        #=======================================================================
        sdx3 = pd.concat([sdx2, pd.concat({'s12R':sdx_s2.subtract(sdx_s1)}, axis=1, names=['base'])], axis=1)
        
        #=======================================================================
        # resid normalized
        #=======================================================================
        """ (s2-s1)/s1"""
        sdx4 = pd.concat([sdx3, pd.concat({'s12Rn':sdx3['s12R'].divide(sdx3['s1'])}, axis=1, names=['base'])], axis=1
                         ).sort_index(sort_remaining=True, axis=1)
        
        sdx4.index.name='dsc'
        
        #switch to hazard order
        sdx5 = sdx4.stack('scale').unstack('dsc')
        
        log.info('constructed residuals w/ %s'%(str(sdx5.shape)))
        #=======================================================================
        # plot residuals
        #=======================================================================
        """ 
        view(sdx5.loc[:, idx[:, 'direct', 'wse', 'mean', 'full']])        
 
        """
        def get_stack(baseName):
            """consistent retrival by base"""            
            #just wd and wse mean (for now)        
            pdx1 = sdx5[baseName].loc[:, idx[:, :, 'mean', :]].droplevel('metric', axis=1) 
            return pdx1.stack(level=pdx1.columns.names).sort_index()
        
        serx = get_stack('s12R')        
        ses.plot_matrix_metric_method_var(serx,
                                          map_d = {'row':'layer','col':'method', 'color':'dsc', 'x':'scale'},
                                          ylab_d={
 
                                              },
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_assets.svg'))
        
        #=======================================================================
        # residuals normalized
        #=======================================================================
        serx = get_stack('s12Rn')        
        ses.plot_matrix_metric_method_var(serx,
                                          map_d = {'row':'layer','col':'method', 'color':'dsc', 'x':'scale'},
                                          ylab_d={
                                            'wd':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$', 
                                            'wse':r'$\frac{\overline{WSE_{s2}}-\overline{WSE_{s1}}}{\overline{WSE_{s1}}}$', 
                                            #'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
                                              },
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd_assets.svg'))
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