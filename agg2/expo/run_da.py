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
        
        matrix_kwargs = dict(figsize=(10,10), set_ax_title=False, add_subfigLabel=False)
        
        #=======================================================================
        # data prep
        #=======================================================================

        
        #get the rsc for each asset and scale        
        dsc_df = pd.read_pickle(fp_lib['direct']['arsc'])
        
        #join the simulation results (and clean up indicides
        samp_dx = ses.join_layer_samps(fp_lib)
        
        """
        samp_dx.loc[:, idx['filter', 'wse', (1, 8)]].hist()
        """
        
        wet_bx = ~samp_dx.loc[:, idx[:, 'wse', :]].isna().all(axis=1)
        #=======================================================================
        # GRANUALR-------
        #=======================================================================
        """agg-stats are pretty nice for the rasters, where domains are consistent
        but for assets, the number exposed varies so much it's more informative to work per-asset"""
        
        kdx_s1 = samp_dx.loc[:, idx[:, :,1]].droplevel('scale', axis=1)#baseline values
        
        #add dummy level
        pdc_kwargs = dict(axis=1, names=['base'])
        kdx1 = pd.concat({'samps':samp_dx}, **pdc_kwargs)
        
        #=======================================================================
        # residuals
        #=======================================================================
        kdx_resid = samp_dx.subtract(kdx_s1)
        
        kdx2 = pd.concat([kdx1, pd.concat({'resid':kdx_resid}, **pdc_kwargs)], axis=1)
        
        #=======================================================================
        # resid.normd
        #=======================================================================
        """any 'inf' values here are false positives"""
        kdx_rnorm = kdx_resid.divide(kdx_s1)
        
        kdx3 = pd.concat([kdx2, pd.concat({'residN':kdx_rnorm}, **pdc_kwargs)], axis=1)        
        
        """        
        view(kdx3[wet_bx].loc[:, idx[:, 'direct', 'wd', (1, 512)]].head(100))
        """
        log.info('compiled granular samples %s for %s'%(str(kdx3.shape), kdx3.columns.unique('base').tolist()))
        
        #=======================================================================
        # zonal stats
        #=======================================================================
        """aggregating granulars by dsc zone
        'samples' base should be the same as aggregated stats below"""
        def get_zonal_stats(gdx, gcols, gkeys):
            """help compute dsc stats then add some dummy levels"""
            dxi = ses.get_dsc_stats1(gdx)   #compute zonal stats
            dxi.columns = append_levels(dxi.columns, dict(zip(gcols, gkeys))) #add the levels back
            return dxi
            
        
        gcols = ['method', 'layer', 'base']
 
        d = dict()
        for i, (gkeys, gdx) in enumerate(kdx3.loc[:, idx[:, :, :, :]].groupby(level=gcols, axis=1)):
            #merge gruoped data w/ dsc 
            gdx_c = pd.concat({gkeys[1]:gdx.droplevel(gcols, axis=1), 'dsc':dsc_df}, axis=1, names=['layer']).sort_index(sort_remaining=True, axis=1)
            
            d[i] = get_zonal_stats(gdx_c, gcols, gkeys)
        
        #merge
        skdx1 = pd.concat(d.values(), axis=1).reorder_levels(['base'] + list(samp_dx.columns.names)+['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
        skdx1.index.name='dsc'
        
        #switch to hazard order
        sdx = skdx1.stack('scale').unstack('dsc')
        log.info('compiled granualr stats on dsc zones %s'%str(sdx.shape))
        """
        view(sdx.T)
        """
        #=======================================================================
        # GRANUALR.PLOTS---------
        #=======================================================================
        row = 'layer_metric'
        def get_stackG(baseName):
            """consistent retrival by base"""            
            
            dx1 = sdx[baseName]
            
            dx2 = dx1.loc[:, idx[:, 'wd', 'mean', :]].join(
                dx1.loc[:, idx[:, 'wse', ('mean', 'count'), :]])
            
            #concat columsn
            mdf1 = dx2.columns.to_frame().reset_index(drop=True)
            
            mdf1[row] = mdf1['layer'].str.cat(mdf1['metric'], sep='_')
            
            dx2.columns = pd.MultiIndex.from_frame(mdf1.drop(columns=['layer', 'metric']))
            
            """
            view(dx2.loc[:, idx[:, :, 'wse_count']].T)
            """
            
            return dx2.stack(level=dx2.columns.names).sort_index().reindex(
                index=['wse_count', 'wse_mean', 'wd_mean'], level=row) 
            
            #===================================================================
            # #just wd and wse mean (for now)        
            # plot_dx = sdx[baseName].loc[:, idx[:, :, 'mean', :]].droplevel('metric', axis=1) 
            # return plot_dx.stack(level=plot_dx.columns.names).sort_index() 
            #===================================================================
        
        
        #=======================================================================
        # samps
        #=======================================================================
        """how does this treat the changing demonminator?
            stats consider the full population at each resolution
            
        
        can we plot just the wet assets?

        """
        #=======================================================================
        # baseName='samps'
        # serx = get_stackG(baseName)        
        # ses.plot_matrix_metric_method_var(serx, 
        #                                   title='asset samples',matrix_kwargs=matrix_kwargs,
        #                                   map_d = {'row':row,'col':'method', 'color':'dsc', 'x':'scale'},
        #                                   ylab_d={'wd':'wd mean', 'wse':'wse mean'},
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_var_gran_%s.svg'%baseName))
        #=======================================================================
        
        #=======================================================================
        # residuals
        #=======================================================================
        """
        wse_count?
            why are these similar between the two?
            why is the full so high?
            
        wse_mean:direct:DD
            why so squirly?
        """
        
        baseName='resid'
        serx = get_stackG(baseName)        
        ses.plot_matrix_metric_method_var(serx,
                                          title='asset sample residuals',matrix_kwargs=matrix_kwargs,
                                          map_d = {'row':row,'col':'method', 'color':'dsc', 'x':'scale'},
                                          ylab_d={'wd':'wd mean resid', 'wse':'wse mean resid'},
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_gran_%s.svg'%baseName))
        
        return
        
        #=======================================================================
        # AGG----------
        #=======================================================================
        log.info('\n\nAGGREGATED PLOTS\n\n')
        #=======================================================================
        # #compute s2 zonal
        #=======================================================================
        gcols = ['method', 'layer']
        d = dict()
        for i, (gkeys, gdx) in enumerate(samp_dx.groupby(level=gcols, axis=1)):
            gdx_c = pd.concat({gkeys[1]:gdx.droplevel(gcols, axis=1), 'dsc':dsc_df}, axis=1, names=['layer']).sort_index(sort_remaining=True, axis=1)
            d[i] = get_zonal_stats(gdx_c, gcols, gkeys)
         
        #merge
        adx_s2 = pd.concat(d.values(), axis=1).reorder_levels(list(samp_dx.columns.names)+['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
         
        adx1 = pd.concat({'s2':adx_s2}, axis=1, names=['base'])
        log.info('built s2 zonal w/ %s'%(str(adx1.shape)))
        #=======================================================================
        # s1 zonal
        #=======================================================================
        d = dict()
        for i, (gkeys, gdx) in enumerate(samp_dx.groupby(level=gcols, axis=1)):
            #always computing against the fine samples
            gdx0 = pd.concat({res:gdx.droplevel(gcols, axis=1).iloc[:, 0] for res in gdx.columns.unique('scale')}, axis=1, names='scale')
 

            gdx_c = pd.concat({gkeys[1]:gdx0, 'dsc':dsc_df}, axis=1, names=['layer']).sort_index(sort_remaining=True, axis=1)
            d[i] = get_zonal_stats(gdx_c, gcols, gkeys)
        
        #merge
        adx_s1 = pd.concat(d.values(), axis=1).reorder_levels(list(samp_dx.columns.names)+['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
        
        adx2 = pd.concat([adx1, pd.concat({'s1':adx_s1}, axis=1, names=['base'])], axis=1)
        log.info('built s1 zonal w/ %s'%(str(adx2.shape)))       
        #=======================================================================
        # #compute residual
        #=======================================================================
        adx3 = pd.concat([adx2, pd.concat({'s12R':adx_s2.subtract(adx_s1)}, axis=1, names=['base'])], axis=1)
        
        #=======================================================================
        # resid normalized
        #=======================================================================
        """ (s2-s1)/s1"""
        adx4 = pd.concat([adx3, pd.concat({'s12Rn':adx3['s12R'].divide(adx3['s1'])}, axis=1, names=['base'])], axis=1
                         ).sort_index(sort_remaining=True, axis=1)
        
        adx4.index.name='dsc'
        
        #switch to hazard order
        adx5 = adx4.stack('scale').unstack('dsc')
        
        log.info('constructed residuals w/ %s'%(str(adx5.shape)))
        
     

        #=======================================================================
        # AGG.PLOTS-------
        #=======================================================================
        
        
        def get_stackA(baseName):
            """consistent retrival by base"""            
            #just wd and wse mean (for now)        
            plot_dx = adx5[baseName].loc[:, idx[:, :, 'mean', :]].droplevel('metric', axis=1) 
            return plot_dx.stack(level=plot_dx.columns.names).sort_index() 
        
        #=======================================================================
        # samples  
        #=======================================================================
        baseName='s2'
        serx = get_stackA(baseName)        
        ses.plot_matrix_metric_method_var(serx,
                                          map_d = {'row':'layer','col':'method', 'color':'dsc', 'x':'scale'},
                                          ylab_d={},
                                          matrix_kwargs=matrix_kwargs,
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_assets_agg_%s.svg'%baseName))
        #=======================================================================
        # residuals
        #=======================================================================        
        """ 
        wd:direct?
            seems like these should be flat... like the rasters
            
            the issue is the FalsePositives... I think we want to work with deltas
        
        view(sdx5.loc[:, idx[:, 'direct', 'wse', 'mean', 'full']])        
 
        """
        baseName='s12R'
        serx = get_stackA(baseName)        
        ses.plot_matrix_metric_method_var(serx,
                                          map_d = {'row':'layer','col':'method', 'color':'dsc', 'x':'scale'},
                                          ylab_d={},
                                          matrix_kwargs=matrix_kwargs,
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_assets_agg_%s.svg'%baseName))
        
        #=======================================================================
        # residuals normalized
        #=======================================================================
        #=======================================================================
        # serx = get_stack('s12Rn')        
        # ses.plot_matrix_metric_method_var(serx,
        #                                   map_d = {'row':'layer','col':'method', 'color':'dsc', 'x':'scale'},
        #                                   ylab_d={
        #                                     'wd':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$', 
        #                                     'wse':r'$\frac{\overline{WSE_{s2}}-\overline{WSE_{s1}}}{\overline{WSE_{s1}}}$', 
        #                                     #'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
        #                                       },
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_var_resid_normd_assets.svg'))
        #=======================================================================
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