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
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
idx = pd.IndexSlice
import matplotlib.pyplot as plt

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
        #see haz.run_da
        haz_da_fp = r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\da\20220919\SJ_r8_haz_da_0919_aggStats_dx.pkl',
        run_name='r8'):
    return run_plots(fp_lib, run_name=run_name, haz_da_fp=haz_da_fp)

def run_plots(fp_lib,haz_da_fp=None,
                  **kwargs):
    """construct figure from SJ downscale cat results"""
    from agg2.expo.da import ExpoDASession as Session
 
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(fp_lib['direct']['catMasks'])).parents[3], #C:/LS/10_OUT/2112_Agg/outs/agg2/r5
        'da', today_str)
    print('out_dir:   %s'%out_dir)
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, **kwargs) as ses:
        log = ses.logger
        
 
        pdc_kwargs = dict(axis=1, names=['base'])
        matrix_kwargs = dict(figsize=(10,10), set_ax_title=False, add_subfigLabel=False)
        lvls = ['base', 'method', 'layer', 'metric', 'dsc']
        
        def sort_dx(dx):
            return dx.reorder_levels(lvls, axis=1).sort_index(axis=1, sort_remaining=True)
        #=======================================================================
        # data prep
        #=======================================================================
        
        # get the rsc for each asset and scale        
        dsc_df = pd.read_pickle(fp_lib['direct']['arsc'])
        
        # join the simulation results (and clean up indicides
        samp_dx_raw = ses.join_layer_samps(fp_lib)
        
        """
        samp_dx.loc[:, idx['filter', 'wse', (1, 8)]].hist()
        """
        
        #=======================================================================
        # compute exposures
        #=======================================================================
        wet_bx = ~samp_dx_raw.loc[:, idx[:, 'wse',:]].isna().all(axis=1)
        wet_bxcol = samp_dx_raw.loc[:, idx[:, 'wse',:]].droplevel('layer', axis=1).notna().astype(int)
        
        #check false negatives        
        wet_falseNegatives = (wet_bxcol.subtract(wet_bxcol.loc[:, idx[:, 1]].droplevel('scale', axis=1))==-1).sum()
        if wet_falseNegatives.any():
            log.info('got %i False Negative exposures\n%s'%(
                wet_falseNegatives.sum(), wet_falseNegatives[wet_falseNegatives!=0]))
        
        #merge
        samp_dx = pd.concat([samp_dx_raw, pd.concat({'expo':wet_bxcol}, axis=1 ,names=['layer']).swaplevel('layer', 'method', axis=1)], 
                            axis=1).sort_index(axis=1, sort_remaining=True)
                            
        
        #=======================================================================
        # GRANULAR (raster style)-------
        #=======================================================================
        """
        unlike Haz, we are computing the stats during data analysis
        
        s2 is computed against the matching samples
        s1 is computed against the baseline samples"""
        
        samp_base_dx = samp_dx.loc[:, idx[:, :, 1]].droplevel('scale', axis=1)
        ufunc_d = {'expo':'sum', 'wd':'mean', 'wse':'mean'}
        
         
        #get baseline stats
        d = dict()
        for layName, stat in ufunc_d.items():
            dxi = samp_base_dx.loc[:, idx[:, layName]].droplevel('layer', axis=1)
            d[layName] = pd.concat({stat:getattr(dxi, stat)()}, axis=1, names='metric')
        
        s1_sdxi = pd.concat(d, axis=1, names=['layer']).unstack().rename(1).to_frame().T.rename_axis('scale')
        s1_sdx = sort_dx(pd.concat({'full':pd.concat({'s1':s1_sdxi, 's2':s1_sdxi}, axis=1, names=['base'])},axis=1, names=['dsc']))
 
        #sampled stats
        res_d=dict()
        for scale, col in dsc_df.items():
            """loop and compute stats with different masks"""
            d = dict()
            #===================================================================
            # #s1
            #===================================================================
            mdex = pd.MultiIndex.from_frame(samp_base_dx.index.to_frame().reset_index(drop=True).join(col.rename('dsc')))
            s1_dx = pd.DataFrame(samp_base_dx.values, index=mdex, columns=samp_base_dx.columns)
            
            d['s1'] = ses.get_dsc_stats2(s1_dx, ufunc_d=ufunc_d)
            #===================================================================
            # s2
            #===================================================================
            s2_dx = pd.DataFrame(samp_dx.loc[:, idx[:, :, scale]].values, index=mdex, columns=samp_base_dx.columns)
            
            d['s2'] = ses.get_dsc_stats2(s2_dx, ufunc_d=ufunc_d)
            
            #JOIN
            res_d[scale] = pd.concat(d, axis=1, names=['base'])
            
        #wrap
        sdx1 = sort_dx(pd.concat(res_d, axis=1, names=['scale']).stack('scale').unstack('dsc'))
        
        sdx2 = pd.concat([sdx1, s1_sdx]).sort_index() #add reso1
        #=======================================================================
        # compute residuals
        #=======================================================================
        srdx = sdx1['s2'].subtract(sdx1['s1'])
        
        
        #normed
        srdxN = srdx.divide(sdx1['s1'])
        
        sdx2 = pd.concat([sdx1, pd.concat({'s12':srdx, 's12N':srdxN}, **pdc_kwargs)], axis=1)
        
        #=======================================================================
        # GRANULAR.PLOT---------
        #=======================================================================
        row = 'layer'
        def get_stackG(baseName):
            """consistent retrival by base"""            
            
            dx1 = sdx2[baseName]
            
            """just 1 metric per layer now"""
            dx2 = dx1.droplevel('metric', axis=1) 
 
            
            """
            view(dx2.loc[:, idx[:, :, 'wse_count']].T)
            """
            
            order_l = {
                'layer_metric':['wse_count', 'wse_mean', 'wd_mean'],
                'layer':['wd', 'wse', 'expo'],
                }[row]
            
            return dx2.stack(level=dx2.columns.names).sort_index().reindex(
                index=order_l, level=row) 
        #=======================================================================
        # plot all
        #=======================================================================
        """how does this treat the changing demonminator?
            stats consider the full population at each resolution
            
        direct:wd
            why are asset samples going up but the rasters are flat?
                rasters are flat by definition
                WW: as se aggregate, only the deepest exposures remain
                
                
             
         
        can we plot just the wet assets?
            might help figure out why the raster averages are the same
            
        normalizing doesn't seem so useful here
        
        plot TP/FP?
 
        """
        #=======================================================================
        # for baseName in ['s1', 's2',  's12N']:
        #     serx = get_stackG(baseName)        
        #     ses.plot_matrix_metric_method_var(serx, 
        #                                       title=baseName,
        #                                       matrix_kwargs=matrix_kwargs,
        #                                       map_d = {'row':row,'col':'method', 'color':'dsc', 'x':'scale'},
        #                                       ylab_d={'wd':'wd mean', 'wse':'wse mean', 'expo':'expo count'},
        #                                       ofp=os.path.join(ses.out_dir, 'metric_method_var_asset_gran_%s.svg'%baseName))
        #=======================================================================
            
        #=======================================================================
        # s12
        #=======================================================================
 #==============================================================================
 #        baseName='s12'
 #        serx = get_stackG(baseName)        
 #        ses.plot_matrix_metric_method_var(serx, 
 #                                          #title=baseName,
 # 
 #                                          map_d = {'row':row,'col':'method', 'color':'dsc', 'x':'scale'},
 #                                          ylab_d={
 #                                              'wd':r'$\overline{WSH_{s2}}-\overline{WSH_{s1}}$', 
 #                                              'wse':r'$\overline{WSE_{s2}}-\overline{WSE_{s1}}$', 
 #                                              'expo':r'$\sum WET_{s2} - \sum WET_{s1}$'},
 #                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_asset_gran_%s.svg'%baseName),
 #                                          matrix_kwargs = dict(figsize=(6.5,6.75), set_ax_title=False, add_subfigLabel=True),
 #                                          ax_lims_d = {'y':{
 #                                              'wd':(-0.2,3.0), 'wse':(-1, 10.0), #'expo':(-10, 4000)
 #                                              }},
 #                                          yfmt_func= lambda x,p:'%.1f'%x,
 #                                          legend_kwargs=dict(loc=7)
 #                                          )
 #==============================================================================
        
        #=======================================================================
        # combined w/ raster----
        #=======================================================================
        if not haz_da_fp is None:
            haz_dx = pd.read_pickle(haz_da_fp)
        
 
        return
#===============================================================================
#         #=======================================================================
#         # residuals
#         #=======================================================================
#         """
#         wse_count?
#             filter
#                 shouldn't full always be on top?
#                     pretty close... false negatives can move this around though?
#              
#         wse_mean:direct:DD
#             why so squirly?
#         """
#          
#         baseName='resid'
#         serx = get_stackG(baseName)        
#         ses.plot_matrix_metric_method_var(serx,
#                                           title='asset sample residuals',matrix_kwargs=matrix_kwargs,
#                                           map_d = {'row':row,'col':'method', 'color':'dsc', 'x':'scale'},
#                                           ylab_d={'wd':'wd mean resid', 'wse':'wse mean resid'},
#                                           ofp=os.path.join(ses.out_dir, 'metric_method_var_gran_%s.svg'%baseName))
#         #=======================================================================
#         # GRANUALR-------
#         #=======================================================================
#         """agg-stats compares stats between the base group to the upsampled group
#          
#         this is not great, as we want to attribute bias to regions
#         
#         
#             resid = stat[i=dsc@j_samp, j=j_samp] - stat[i=dsc@j_samp, j=j_base]
#             
#             residN = resid/stat[i=dsc@j_samp, j=j_base]
#             
#         """
#         
#         kdx_s1 = samp_dx.loc[:, idx[:, :,1]].droplevel('scale', axis=1)#baseline values
#         
#         #add dummy level
#         
#         kdx1 = pd.concat({'samps':samp_dx}, **pdc_kwargs)
#         
#         #=======================================================================
#         # residuals
#         #=======================================================================
#         kdx_resid = samp_dx.subtract(kdx_s1).round(4)
#         
#         kdx2 = pd.concat([kdx1, pd.concat({'resid':kdx_resid}, **pdc_kwargs)], axis=1)
#         
#         """
#         kdx2.columns.names
# 
#         dx = kdx2.loc[:, idx[:, 'direct', 'wse', (1, 512)]].head(100).sort_index(axis=1)
#         view(dx)
#         """
#  
#         #=======================================================================
#         # resid.normd
#         #=======================================================================
#         kdx3 = kdx2
#         """no... this corrupts the stats"""
#         #=======================================================================
#         # """any 'inf' values here are false positives"""
#         # kdx_rnorm = kdx_resid.divide(kdx_s1)
#         # 
#         # kdx3 = pd.concat([kdx2, pd.concat({'residN':kdx_rnorm}, **pdc_kwargs)], axis=1)        
#         #=======================================================================
#         
#         """        
#         view(kdx3[wet_bx].loc[:, idx[:, 'direct', 'wd', (1, 512)]].head(100))
#         """
#         log.info('compiled granular samples %s for %s'%(str(kdx3.shape), kdx3.columns.unique('base').tolist()))
#         
#         #=======================================================================
#         # zonal stats
#         #=======================================================================
#         """aggregating granulars by dsc zone
#         'samples' base should be the same as aggregated stats below"""
#         def get_zonal_stats(gdx, gcols, gkeys, **kwargs):
#             """help compute dsc stats then add some dummy levels"""
#             dxi = ses.get_dsc_stats1(gdx, **kwargs)   #compute zonal stats            
#             
#             dxi.columns = append_levels(dxi.columns, dict(zip(gcols, gkeys))) #add the levels back
#             return dxi
#             
#         
#         gcols = ['method', 'layer', 'base']
#  
#         d = dict()
#         print(kdx3.columns.names)
#         for i, (gkeys, gdx) in enumerate(kdx3.loc[:, idx[:, :, :, :]].groupby(level=gcols, axis=1)):
#             layName = gkeys[1]
#             #merge gruoped data w/ dsc 
#             gdx_c = pd.concat({layName:gdx.droplevel(gcols, axis=1), 'dsc':dsc_df}, axis=1, names=['layer']).sort_index(sort_remaining=True, axis=1)
#             
#             #set stat
#             if layName == 'expo':
#                 stat = 'sum'
#             else:
#                 stat = 'mean'
#             
#             
#             d[i] = get_zonal_stats(gdx_c, gcols, gkeys, ufunc_l=[stat])
#         
#         #merge
#         skdx1 = pd.concat(d.values(), axis=1).reorder_levels(['base'] + list(samp_dx.columns.names)+['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
#         skdx1.index.name='dsc'
#         
#         """
#         view(skdx1['resid'].T)
#         """
#  
#         #=======================================================================
#         # #switch to hazard order
#         #=======================================================================
#         sdx = skdx1.stack('scale').unstack('dsc')
#         log.info('compiled granualr stats on dsc zones %s'%str(sdx.shape))        
#  
#         
#         
#         """
#         view(sdx.T)
#         """
#         #=======================================================================
#         # GRANUALR.PLOTS---------
#         #=======================================================================
#         row = 'layer'
#         def get_stackG(baseName):
#             """consistent retrival by base"""            
#             
#             dx1 = sdx[baseName]
#             
#             """just 1 metric per layer now"""
#             dx2 = dx1.droplevel('metric', axis=1) 
#             #===================================================================
#             # dx2 = dx1.loc[:, idx[:, 'wd', 'mean', :]].join(
#             #     dx1.loc[:, idx[:, 'wse', ('mean', 'count'), :]])
#             # 
#             # #concat columsn
#             # mdf1 = dx2.columns.to_frame().reset_index(drop=True)
#             # 
#             # mdf1[row] = mdf1['layer'].str.cat(mdf1['metric'], sep='_')
#             # 
#             # dx2.columns = pd.MultiIndex.from_frame(mdf1.drop(columns=['layer', 'metric']))
#             #===================================================================
#             
#             """
#             view(dx2.loc[:, idx[:, :, 'wse_count']].T)
#             """
#             
#             order_l = {
#                 'layer_metric':['wse_count', 'wse_mean', 'wd_mean'],
#                 'layer':['wd', 'wse', 'expo'],
#                 }[row]
#             
#             return dx2.stack(level=dx2.columns.names).sort_index().reindex(
#                 index=order_l, level=row) 
#             
#             #===================================================================
#             # #just wd and wse mean (for now)        
#             # plot_dx = sdx[baseName].loc[:, idx[:, :, 'mean', :]].droplevel('metric', axis=1) 
#             # return plot_dx.stack(level=plot_dx.columns.names).sort_index() 
#             #===================================================================
#         
#         
#         #=======================================================================
#         # samps
#         #=======================================================================
#         """how does this treat the changing demonminator?
#             stats consider the full population at each resolution
#             
#         direct:wd
#             why are asset samples going up but the rasters are flat?
#                 rasters are flat by definition
#                 WW: as se aggregate, only the deepest exposures remain
#                 
#                 
#              
#          
#         can we plot just the wet assets?
#             might help figure out why the raster averages are the same
#             
#         plot TP and FPs?
#  
#         """
#         baseName='samps'
#         serx = get_stackG(baseName)        
#         ses.plot_matrix_metric_method_var(serx, 
#                                           title='asset samples',matrix_kwargs=matrix_kwargs,
#                                           map_d = {'row':row,'col':'method', 'color':'dsc', 'x':'scale'},
#                                           ylab_d={'wd':'wd mean', 'wse':'wse mean', 'expo':'expo count'},
#                                           ofp=os.path.join(ses.out_dir, 'metric_method_var_gran_%s.svg'%baseName))
#         
#         return
#          
#         #=======================================================================
#         # residuals
#         #=======================================================================
#         """
#         wse_count?
#             filter
#                 shouldn't full always be on top?
#                     pretty close... false negatives can move this around though?
#              
#         wse_mean:direct:DD
#             why so squirly?
#         """
#          
#         baseName='resid'
#         serx = get_stackG(baseName)        
#         ses.plot_matrix_metric_method_var(serx,
#                                           title='asset sample residuals',matrix_kwargs=matrix_kwargs,
#                                           map_d = {'row':row,'col':'method', 'color':'dsc', 'x':'scale'},
#                                           ylab_d={'wd':'wd mean resid', 'wse':'wse mean resid'},
#                                           ofp=os.path.join(ses.out_dir, 'metric_method_var_gran_%s.svg'%baseName))
#===============================================================================
        
        
        #=======================================================================
        # GRANULAR WET-----------
        #=======================================================================
        kdx3
        #=======================================================================
        # AGG----------
        #=======================================================================
        """compute stats per resolution then compare.
        not so useful because the regions change
        
        
        resid = stat[i=dsc@j_samp, j=j_samp] - stat[i=dsc@j_base, j=j_base]
        """
        
#===============================================================================
#         log.info('\n\nAGGREGATED PLOTS\n\n')
#         #=======================================================================
#         # #compute s2 zonal
#         #=======================================================================
#         gcols = ['method', 'layer']
#         d = dict()
#         for i, (gkeys, gdx) in enumerate(samp_dx.groupby(level=gcols, axis=1)):
#             gdx_c = pd.concat({gkeys[1]:gdx.droplevel(gcols, axis=1), 'dsc':dsc_df}, axis=1, names=['layer']).sort_index(sort_remaining=True, axis=1)
#             d[i] = get_zonal_stats(gdx_c, gcols, gkeys)
#          
#         #merge
#         adx_s2 = pd.concat(d.values(), axis=1).reorder_levels(list(samp_dx.columns.names)+['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
#          
#         adx1 = pd.concat({'s2':adx_s2}, axis=1, names=['base'])
#         log.info('built s2 zonal w/ %s'%(str(adx1.shape)))
#         #=======================================================================
#         # s1 zonal
#         #=======================================================================
#         d = dict()
#         for i, (gkeys, gdx) in enumerate(samp_dx.groupby(level=gcols, axis=1)):
#             #always computing against the fine samples
#             gdx0 = pd.concat({res:gdx.droplevel(gcols, axis=1).iloc[:, 0] for res in gdx.columns.unique('scale')}, axis=1, names='scale')
#  
# 
#             gdx_c = pd.concat({gkeys[1]:gdx0, 'dsc':dsc_df}, axis=1, names=['layer']).sort_index(sort_remaining=True, axis=1)
#             d[i] = get_zonal_stats(gdx_c, gcols, gkeys)
#         
#         #merge
#         adx_s1 = pd.concat(d.values(), axis=1).reorder_levels(list(samp_dx.columns.names)+['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
#         
#         adx2 = pd.concat([adx1, pd.concat({'s1':adx_s1}, axis=1, names=['base'])], axis=1)
#         log.info('built s1 zonal w/ %s'%(str(adx2.shape)))       
#         #=======================================================================
#         # #compute residual
#         #=======================================================================
#         adx3 = pd.concat([adx2, pd.concat({'s12R':adx_s2.subtract(adx_s1)}, axis=1, names=['base'])], axis=1)
#         
#         #=======================================================================
#         # resid normalized
#         #=======================================================================
#         """ (s2-s1)/s1"""
#         adx4 = pd.concat([adx3, pd.concat({'s12Rn':adx3['s12R'].divide(adx3['s1'])}, axis=1, names=['base'])], axis=1
#                          ).sort_index(sort_remaining=True, axis=1)
#         
#         adx4.index.name='dsc'
#         
#         #switch to hazard order
#         adx5 = adx4.stack('scale').unstack('dsc')
#         
#         log.info('constructed residuals w/ %s'%(str(adx5.shape)))
#         
#      
# 
#         #=======================================================================
#         # AGG.PLOTS-------
#         #=======================================================================
#         
#         
#         def get_stackA(baseName):
#             """consistent retrival by base"""            
#             #just wd and wse mean (for now)        
#             plot_dx = adx5[baseName].loc[:, idx[:, :, 'mean', :]].droplevel('metric', axis=1) 
#             return plot_dx.stack(level=plot_dx.columns.names).sort_index() 
#         
#         #=======================================================================
#         # samples  
#         #=======================================================================
#         baseName='s2'
#         serx = get_stackA(baseName)        
#         ses.plot_matrix_metric_method_var(serx,
#                                           map_d = {'row':'layer','col':'method', 'color':'dsc', 'x':'scale'},
#                                           ylab_d={},
#                                           matrix_kwargs=matrix_kwargs,
#                                           ofp=os.path.join(ses.out_dir, 'metric_method_var_assets_agg_%s.svg'%baseName))
#         #=======================================================================
#         # residuals
#         #=======================================================================        
#         """ 
#         wd:direct?
#             seems like these should be flat... like the rasters
#             
#             the issue is the FalsePositives... I think we want to work with deltas
#         
#         view(sdx5.loc[:, idx[:, 'direct', 'wse', 'mean', 'full']])        
#  
#         """
#         baseName='s12R'
#         serx = get_stackA(baseName)        
#         ses.plot_matrix_metric_method_var(serx,
#                                           map_d = {'row':'layer','col':'method', 'color':'dsc', 'x':'scale'},
#                                           ylab_d={},
#                                           matrix_kwargs=matrix_kwargs,
#                                           ofp=os.path.join(ses.out_dir, 'metric_method_var_assets_agg_%s.svg'%baseName))
#===============================================================================
        
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
