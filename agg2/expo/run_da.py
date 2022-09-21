'''
Created on Sep. 10, 2022

@author: cefect

exposure data analysis
'''
 

import os, pathlib, itertools, logging, sys
from definitions import proj_lib
from hp.basic import get_dict_str, today_str, lib_iter
from hp.pd import append_levels, view
import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
idx = pd.IndexSlice
import matplotlib.pyplot as plt
import matplotlib


logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )

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
    return run_plots(fp_lib, run_name=run_name)

def run_plots(fp_lib,write=False, **kwargs):
    """construct figure from SJ expo results"""
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
    with Session(out_dir=out_dir, logger=logging.getLogger('run_da'), **kwargs) as ses:
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
        srdx = sdx2['s2'].subtract(sdx2['s1'])
        
        
        #normed
        srdxN = srdx.divide(sdx2['s1'])
        
        sdx3 = pd.concat([sdx2, pd.concat({'s12':srdx, 's12N':srdxN}, **pdc_kwargs)], axis=1)
        
        #=======================================================================
        # write
        #=======================================================================
        if write:
            ofp = os.path.join(ses.out_dir, f'{ses.fancy_name}_aggStats_dx.pkl')
            sdx3.to_pickle(ofp)
            
            log.info(f'wrote {str(sdx3.shape)} to \n    {ofp}')
        
 
        #=======================================================================
        # GRANULAR.PLOT---------
        #=======================================================================
        row = 'layer'
        def get_stackG(baseName):
            """consistent retrival by base"""            
            
            dx1 = sdx3[baseName]
            
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
        baseName='s12'
        serx = get_stackG(baseName)        
        ses.plot_matrix_metric_method_var(serx, 
                                          #title=baseName,
  
                                          map_d = {'row':row,'col':'method', 'color':'dsc', 'x':'scale'},
                                          ylab_d={
                                              'wd':r'$\overline{WSH_{s2}}-\overline{WSH_{s1}}$', 
                                              'wse':r'$\overline{WSE_{s2}}-\overline{WSE_{s1}}$', 
                                              'expo':r'$\sum WET_{s2} - \sum WET_{s1}$'},
                                          ofp=os.path.join(ses.out_dir, 'metric_method_var_asset_gran_%s.svg'%baseName),
                                          matrix_kwargs = dict(figsize=(6.5,6.75), set_ax_title=False, add_subfigLabel=True),
                                          ax_lims_d = {'y':{
                                              'wd':(-0.2,3.0), 'wse':(-1, 10.0), #'expo':(-10, 4000)
                                              }},
                                          yfmt_func= lambda x,p:'%.1f'%x,
                                          legend_kwargs=dict(loc=7)
                                          )
        
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
        

def SJ_combine_plots_0919(
        fp_lib = {
            'haz':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\da\20220919\SJ_r8_haz_da_0919_aggStats_dx.pkl',
            'exp':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\da\20220919\SJ_r8_expo_da_0919_aggStats_dx.pkl'}
        
        ):
    return  run_plots_combine(fp_lib, run_name='r8')


def run_plots_combine(fp_lib,**kwargs):
    """da and figures which combine hazard and exposure
    
    
    
    """
    from agg2.expo.da import ExpoDASession as Session
 
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    out_dir = os.path.join(pathlib.Path(os.path.dirname(fp_lib['exp'])).parents[1],'da', 'expo',today_str)
    print('out_dir:   %s'%out_dir)
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir,logger=logging.getLogger('run_da'),  **kwargs) as ses:
        log = ses.logger 
        
        #=======================================================================
        # load data-------
        #=======================================================================
        haz_dx = pd.read_pickle(fp_lib['haz'])
        log.info(f'loaded hazard data {str(haz_dx.shape)} w/ coldex:\n    {haz_dx.columns.names}')
        
        expo_dx = pd.read_pickle(fp_lib['exp'])
        log.info(f'loaded expo data {str(expo_dx.shape)} w/ coldex:\n    {expo_dx.columns.names}')
        
        
        #===================================================================
        # check consistency
        #===================================================================
        assert np.array_equal(
            haz_dx.index.to_frame().reset_index(drop=True)['scale'].values,
            expo_dx.index.values
            )
        
        assert len(set(haz_dx.columns.names).symmetric_difference(expo_dx.columns.names))==0, 'column name mismatch'
        
        #chekc column values
        hmdex, amdex = haz_dx.columns, expo_dx.columns
        for aname in ['base', 'method', 'dsc']:
            assert np.array_equal(
                hmdex.unique(aname),
                amdex.unique(aname)
                ), aname
        
        assert set(hmdex.unique('layer')).difference(amdex.unique('layer'))==set(), 'layer name mismatch'
        
        #===================================================================
        # join
        #===================================================================
        haz_dx1 = haz_dx.reorder_levels(expo_dx.columns.names, axis=1).droplevel((1,2))
        dx1 = pd.concat({'exp':expo_dx, 'haz':haz_dx1},names=['phase'], axis=1)
        
        log.info(f'merged haz and expo data to get {str(dx1.shape)} w/ coldex:\n    {dx1.columns.names}')
        
        #=======================================================================
        # separate data------
        #=======================================================================
        """index slicer for each axis"""
        mdex=dx1.columns
        
        #['direct_haz', 'direct_exp', 'filter_haz', 'filter_exp']
        col_keys = ['_'.join(e) for e in list(itertools.product(mdex.unique('method').values, ['haz', 'exp']))]
        row_keys = ['wd_bias', 'wse_error', 'exp_area', 'vol']
 
        #empty container
        idx_d = dict()
        post_lib=dict()
        
 
        
        def set_row(method, phase):
            coln = f'{method}_{phase}'
            assert not coln in idx_d
            idx_d[coln] = dict()
            post_lib[coln] = dict()
            return coln
        #=======================================================================
        # #direct_haz
        #=======================================================================
        ea_base = 's12N'
        method, phase = 'direct', 'haz'
        
        coln = set_row(method, phase)        
        idx_d[coln]['wd_bias'] =    idx[phase, 's12N', method,'wd', 'mean', :]
        idx_d[coln]['wse_error'] =  idx[phase, 's12', method,'wse', 'mean', :]
        idx_d[coln]['exp_area'] =   idx[phase, ea_base, method,'wd', 'posi_area', :]
        idx_d[coln]['vol'] =        idx[phase, 's12N', method,'wd', 'vol', :]        
 
        #=======================================================================
        # direct_exp
        #=======================================================================
        method, phase = 'direct', 'exp'
        
        coln = set_row(method, phase)
        idx_d[coln]['wd_bias'] =    idx[phase, 's12N', method,'wd', 'mean', :]
        idx_d[coln]['wse_error'] =  idx[phase, 's12', method,'wse', 'mean', :]
        idx_d[coln]['exp_area']=    idx[phase, ea_base, method,'expo', 'sum', :]
        #idx_d[coln]['vol'] =        idx[phase, 's12N', method,'wd', 'vol', :]
        
        #=======================================================================
        # #filter_haz
        #=======================================================================
        method, phase = 'filter', 'haz'
        
        coln = set_row(method, phase)        
        idx_d[coln]['wd_bias'] =    idx[phase, 's12N', method,'wd', 'mean', :]
        idx_d[coln]['wse_error'] =  idx[phase, 's12', method,'wse', 'mean', :]
        idx_d[coln]['exp_area'] =   idx[phase, ea_base, method,'wd', 'posi_area', :]
        idx_d[coln]['vol'] =        idx[phase, 's12N', method,'wd', 'vol', :]
        
        #=======================================================================
        # filter_exp
        #=======================================================================
        method, phase = 'filter', 'exp'
        
        coln = set_row(method, phase)
        idx_d[coln]['wd_bias'] =    idx[phase, 's12N', method,'wd', 'mean', :]
        idx_d[coln]['wse_error'] =  idx[phase, 's12', method,'wse', 'mean', :]
        idx_d[coln]['exp_area'] =   idx[phase, ea_base, method,'expo', 'sum', :]
        #idx_d[coln]['vol'] =        idx[phase, 's12N', method,'wd', 'vol', :]
        
        #=======================================================================
        # check
        #=======================================================================
        cnt = 0
        for colk, d in idx_d.items():
            assert colk in col_keys, colk
            for rowk, idxi in d.items(): 
                assert rowk in row_keys, rowk
                assert len(dx1.loc[:, idxi])>0, f'bad on {rowk}.{colk}'
                
                cnt+=1
                
        log.info('built %i data selectors'%cnt)
        
        #=======================================================================
        # #collect
        #=======================================================================
        data_lib = {c:dict() for c in row_keys} #matching convention of get_matrix_fig() {row_key:{col_key:ax}}
        
        for colk in col_keys:
            for rowk in row_keys:
                if rowk in idx_d[colk]:
                    idxi = idx_d[colk][rowk]
                    data_lib[rowk][colk] = dx1.loc[:, idxi].droplevel(list(range(5)), axis=1)                
            
            
        #===================================================================
        # plot------
        #===================================================================
        """such a custom plot no use in writing a function
        
        WSH:direct
            why is fulldomain flat but exposed increasing?
                full: computes the same metric for reporting as for aggregating. averaging is commutative
                exposed: 
        
        TODO:
        
        """
        ax_d, keys_all_d = ses.plot_grid_d(data_lib, post_lib)
 
        
        
        ylab_d={
              'wd_bias':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$', 
            'wse_error':r'$\overline{WSE_{s2}}-\overline{WSE_{s1}}$', 
              'exp_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
              'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$',
              }
        
        ax_title_d = {
            'direct_haz': 'full domain', 
            'direct_exp':'exposed domain',
             'filter_haz':'full domain',
             'filter_exp':'exposed domain',            
            }
        
        ax_ylims_d = {
              'wd_bias':(-2,10), 
              'wse_error':(-1,15), 
              #'exp_area':(-1,20),
              'vol':(-0.2, .01),
            }
        
        ax_yprec_d = {
              'wd_bias':0, 
              'wse_error':1, 
              'exp_area':0,
              'vol':1,
            }
        
        
        for row_key, col_key, ax in lib_iter(ax_d):
            ax.grid()
            
            #first col
            if col_key == col_keys[0]:
                ax.set_ylabel(ylab_d[row_key])
                if row_key in ax_ylims_d:                
                    ax.set_ylim(ax_ylims_d[row_key])
                digits = ax_yprec_d[row_key]
                """not working... settig on all
                ax.yaxis.set_major_formatter(lambda x,p:f'%.{digits}f'%x)
                #ax.get_yaxis().set_major_formatter(lambda x,p:'{0:.{1}}'.format(x, digits))"""
                
                #last row
                if row_key==row_keys[-1]:
                    pass
                
            #first row
            if row_key==row_keys[0]:
                ax.set_title(ax_title_d[col_key])
                #ax.set_yscale('log')
                
                #last col
                if col_key==col_keys[-1]:
                    ax.legend()
                
                
            #last row
            if row_key==row_keys[-1]:
                ax.set_xlabel('resolution (m)')
                
                if 'exp' in col_key:
                    ax.axis('off')
                    for txt in ax.texts:
                        txt.set_visible(False)
 
        
        #add the titles
        fig = ax.figure
        fig.suptitle('direct', x=0.32)        
        fig.text(0.8, 0.98, 'filter and subtract', size=matplotlib.rcParams['figure.titlesize'], ha='center')
        #fig.suptitle('filter and subtract', x=0.8)
        #=======================================================================
        # output
        #=======================================================================
        ofp = os.path.join(ses.out_dir, f'{ses.fancy_name}_matrix_combined.svg')
        ses.output_fig(fig, ofp=ofp)
        """
        plt.show()
        """
 
        
if __name__ == "__main__":
    SJ_plots_0910()
    #SJ_combine_plots_0919()
