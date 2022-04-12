'''
Created on Feb. 21, 2022

@author: cefect

analysis on hyd.model outputs
'''
#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
start = datetime.datetime.now()
print('start at %s' % start)

import qgis.core
import pandas as pd
import numpy as np


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
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)


from agg.hyd.analy.analy_scripts import ModelAnalysis

 

def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r2',
        overwrite=True,
 
        #=======================================================================
        # #data
        #=======================================================================
        catalog_fp = None,
        modelID_l=None,
        baseID = 0, #model representing the base run
 
        #=======================================================================
        # plot control
        #=======================================================================
        transparent=False,
        
        #=======================================================================
        # debugging
        #=======================================================================
 
        
        **kwargs):
    
    with ModelAnalysis(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                       catalog_fp=catalog_fp,baseID=baseID,modelID_l=modelID_l,
                       bk_lib = {
                           'outs':dict(),
                           'finv_agg_fps':dict(),
                           'trues':dict(baseID=baseID),
                           
                           },
                 **kwargs) as ses:
        
        ses.runCompileSuite()
 
        
        
        
        #=======================================================================
        # individual model summaries---------
        #=======================================================================
        mids = [6]
        #ses.write_resvlay(dkey='rsamps', modelID_l=mids)
        #=======================================================================
        # for mid in mids:
        #     ses.plot_model_smry(mid)
        #=======================================================================
        
        #=======================================================================
        # total loss (top level) summary bar charts--------
        #=======================================================================
        
        #ses.write_suite_smry()
        
        #tloss sum: agg levels  X studyArea (w/ tval_samp=rand)
        mids = [11, 19, 20]
        #ses.plot_total_bars(modelID_l=mids, plot_bgrp='aggLevel')
        
        #hazard res X study area (bars for aggLevel)
        mids = list(range(9))
        #ses.plot_total_bars(modelID_l=mids, dkey_d={'tloss':'sum'}, plot_bgrp='aggLevel',  plot_rown='resolution', plot_coln='studyArea', sharey='col')
        
        #haz res x aggLevel (tloss bars for studyArea)
        #=======================================================================
        # ses.plot_total_bars(modelID_l=mids, dkey_d={'tloss':'sum'}, plot_bgrp='studyArea',  
        #                     plot_rown='aggLevel', plot_coln='resolution', sharey='row', baseline_loc='first_axis')
        #=======================================================================
            
        #=======================================================================
        # hazard data----------
        #=======================================================================
        

        
        #=======================================================================
        # resolution vs. studyArea
        #=======================================================================
        mids = [0, 1, 2]
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=mids, plot_type='scatter',
        #                      plot_rown='studyArea', plot_coln='resolution', fmt='png',
        #                      sharey='row', sharex='row')
        #=======================================================================
        #=======================================================================
        # resolution vs. dsampStage
        #=======================================================================
        mids=[0,1,2,21,22] 
        #all study areas
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=mids, plot_type='scatter',
        #                      plot_rown='dsampStage', plot_coln='resolution', fmt='png',
        #                      sharey='all', sharex='all')
        #=======================================================================
        
        #per-study area (just dsampStage=depth)
        mids=[0,21,22] 
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=mids, plot_type='scatter',
        #                      plot_rown='studyArea', plot_coln='resolution', fmt='png',
        #                      sharey='row', sharex='row')
        #=======================================================================
        
        #=======================================================================
        # resolution vs downSampling method
        #=======================================================================
        mids=[0,1,2,23,24]
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=mids, plot_type='scatter',
        #                      plot_rown='downSampling', plot_coln='resolution', fmt='png',
        #                      sharey='all', sharex='all')
        #=======================================================================
        
        #per-study area (just dsampStage=depth)
        mids=[0,23,24] 
        #=======================================================================
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=mids, plot_type='scatter',
        #                      plot_rown='studyArea', plot_coln='resolution', fmt='png',
        #                      sharey='row', sharex='row')
        #=======================================================================
        
        #=======================================================================
        # asset data----------
        #=======================================================================
        
        #=======================================================================
        # aggLevel vs. studyArea (rsamps)
        #=======================================================================
        mids = [0, 3,6]
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=mids, plot_type='scatter',
        #                      plot_rown='studyArea', plot_coln='aggLevel', fmt='png',
        #                      sharey='row', sharex='row')
        #=======================================================================
        
        #aggType=convexHull
        mids = [0, 25,26]
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=mids, plot_type='scatter',
        #                      plot_rown='studyArea', plot_coln='aggLevel', fmt='png',
        #                      sharey='row', sharex='row')
        #=======================================================================
        
        #=======================================================================
        # aggLevel vs. aggType (rsamps)
        #=======================================================================
        """This one is tricky. these are not realy comparable....
        total coount per study area? total asset area per study area"""

        #=======================================================================
        # tval_type vs. aggLevel (tvals)
        #=======================================================================
        mids = [0,3,6,    #tval_type=uniform
                11,19,20, #tval_type=rand  
                29,30,31, #tval_type=footprintArea

                      ]
        
        """no... there is no true value to compare to
        ses.plot_compare_mat(dkey='tvals', modelID_l=mids, plot_type='scatter',
                             plot_rown='tval_type', plot_coln='aggLevel', fmt='png',
                             sharey='all', sharex='all')"""
                
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='tvals', plot_type='violin', 
        #                   plot_rown='tval_type', plot_coln='aggLevel', plot_colr='aggLevel',
        #                   sharey='all', sharex='all')
        #=======================================================================
        
        """could also be nice to plot vs. polygon area?"""
        
        #=======================================================================
        # Asset Value Attribution (dscale_meth vs aggLevel)
        #=======================================================================
        
        mids = [0,
                3,6, #dscale_meth=centroid, 
                27,28, #dscale_meth=area_split 
                ]
        
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='tvals', plot_type='hist', 
        #                   plot_rown='dscale_meth', plot_coln='aggLevel', plot_colr='aggLevel',
        #                   sharey='all', sharex='all')
        #=======================================================================
        
        #=======================================================================
        # Asset Value Attribution (dscale_meth vs tval_type)
        #=======================================================================
        mids = [ 
                #tval_type=uniform
                6, #dscale_meth=centroid, 
                28, #dscale_meth=area_split
                #tval_type=rand
                20, #dscale_meth=centroid, 
                32, #dscale_meth=area_split
                #tval_type=footprintArea
                31, #dscale_meth=centroid, 
                33, #dscale_meth=area_split
                ]
        
 
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='tvals', plot_type='violin', 
        #                   plot_rown='dscale_meth', plot_coln='tval_type', plot_colr='tval_type',
        #                   sharey='all', sharex='all')
        #=======================================================================
        
 
        #=======================================================================
        # Intersection-----------
        #=======================================================================
 
        
        #=======================================================================
        # #hazard vs asset resolution (depths)
        #=======================================================================
        #simple
        fmt='png'
        #ses.plot_dkey_mat(dkey='rsamps', modelID_l=list(range(9)), plot_rown='aggLevel', plot_coln='resolution',  fmt='svg',sharex='all',sharey='all', plot_colr='aggLevel')
        
        
        for plotName, mids in {
            #'wse':list(range(9)),
            'depth':[0, 21,22, #dsampStage=depth
                     3,34,35,
                     6,36,37,
                     ]}.items(): 
            pass
         
            #main scatter plot       
            ses.plot_compare_mat(dkey='rsamps', modelID_l=mids,plot_rown='aggLevel', plot_coln='resolution', fmt=fmt,sharex='all',sharey='all')
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=[0,3,6], plot_rown='aggLevel', plot_coln='studyArea',  fmt=fmt)
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=[0,1,2], plot_rown='resolution', plot_coln='studyArea', plot_colr='studyArea',  fmt=fmt)
        #=======================================================================
        #=======================================================================
        
        #=======================================================================
        # loss calc: UNIT loss---------
        #=======================================================================


        #=======================================================================
        # hazard vs asset resolution 
        #=======================================================================
        for plotName, mids in {
            #'798':list(range(9)),
            '811':list(range(34, 43))}.items():
 
            ses.plot_compare_mat(dkey='rloss', modelID_l=mids,plot_rown='aggLevel', plot_coln='resolution', fmt='png',
                                     sharex='all',sharey='all')
            
            #histogram of rloss
            #=======================================================================
            # ses.plot_dkey_mat(dkey='rloss', modelID_l=mids,plot_rown='aggLevel', plot_coln='resolution',plot_colr='aggLevel',
            #                          sharex='all',sharey='all')
            #=======================================================================
            
            #just points on the vfunc
            #=======================================================================
            # ses.plot_vs_mat(modelID_l=mids, fmt='png', plot_rown='aggLevel', plot_coln='resolution',
            #                 dkey_y='rloss', dkey_x='rsamps',)
            #=======================================================================
            
        #=======================================================================
        # hazard res vs. studyArea
        #=======================================================================
        mids = list(range(3))
        #=======================================================================
        # ses.plot_compare_mat(dkey='rloss', modelID_l=mids,plot_rown='studyArea', plot_coln='resolution', fmt='png',
        #                          sharex='all',sharey='all')
        #=======================================================================
        

        
        #histogram of rloss results
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='rloss', plot_coln='vid', plot_rown='studyArea', plot_colr='studyArea', 
        #                   plot_type='hist', add_label=True, drop_zeros=True, bins=20,  )
        #=======================================================================
        
        #as a function of aggLevel
        mids = [0,3, 6, 12, 13, 14, 15, 16, 17]
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='rloss', plot_coln='vid', plot_rown='aggLevel', plot_colr='aggLevel', 
        #                   plot_type='hist', add_label=True)
        #=======================================================================
        
        #=======================================================================
        # error scatters
        #=======================================================================
        #hazard vs asset resolution
        #=======================================================================
        # mids = list(range(9))
        # fmt='png'
        # ses.plot_compare_mat(dkey='rloss', modelID_l=mids,plot_rown='aggLevel', plot_coln='resolution', fmt=fmt)
        # ses.plot_compare_mat(dkey='rloss', modelID_l=[0,3,6], plot_rown='aggLevel', plot_coln='studyArea',  fmt=fmt)
        # ses.plot_compare_mat(dkey='rloss', modelID_l=[0,1,2], plot_rown='resolution', plot_coln='studyArea', plot_colr='studyArea',  fmt=fmt)
        #=======================================================================
        
        #=======================================================================
        # loss calc: total loss-------
        #=======================================================================
        
        #=======================================================================
        # input value plots
        #=======================================================================
        mids = list(range(9))
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='tvals', plot_rown='aggLevel', 
        #                   plot_coln='resolution', plot_type='box', drop_zeros=True,
        #                   plot_colr='aggLevel', bins=30,  )
        #=======================================================================
        
        
        #=======================================================================
        # #histogram of tloss results
        #=======================================================================
        #vs. study area
        mids = [0,12,13]
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='tloss', plot_coln='vid', plot_rown='studyArea', plot_colr='studyArea', 
        #                   plot_type='hist', add_label=True, drop_zeros=True, bins=20,  )
        #=======================================================================
        
        #vs aggLevel
        mids = [0,3, 6, 12, 13, 14, 15, 16, 17]
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='tloss', plot_coln='vid', plot_rown='aggLevel', plot_colr='aggLevel', 
        #                   plot_type='hist', add_label=True, xlims=(0,10))
        #=======================================================================
        
        #=======================================================================
        # error scatters
        #=======================================================================
        
        #hazard vs asset resolution
        mids = list(range(9))
        fmt='png'
        #ses.plot_compare_mat(dkey='tloss',aggMethod='sum',modelID_l=mids,plot_rown='aggLevel', plot_coln='resolution', fmt=fmt)
        #ses.plot_compare_mat(dkey='tloss', modelID_l=[0,3,6], plot_rown='aggLevel', plot_coln='studyArea',  fmt=fmt, aggMethod='sum')
        #ses.plot_compare_mat(dkey='tloss', modelID_l=[0,1,2], plot_rown='resolution', plot_coln='studyArea', plot_colr='studyArea',  fmt=fmt, aggMethod='sum')
        
        #=======================================================================
        # errors bars (totals)
        #=======================================================================
        #total errors (relative) grouped by studyArea
        #=======================================================================
        # ses.plot_compare_mat(dkey='tloss',aggMethod='sum',modelID_l=mids, plot_type='bars',
        #                      plot_rown='aggLevel', plot_coln='resolution', plot_bgrp='studyArea',
        #                      err_type='relative' )
        #=======================================================================
        
        
        #=======================================================================
        # errors violin
        #=======================================================================
        #=======================================================================
        # ses.plot_compare_mat(dkey='tloss',aggMethod='sum',modelID_l=mids, plot_type='violin',
        #                      plot_rown='aggLevel', plot_coln='resolution', plot_bgrp='studyArea',
        #                      err_type='relative' )
        # 
        #=======================================================================
        
        #=======================================================================
        # tval_type
        #=======================================================================

        
            

        
        out_dir = ses.out_dir
        
    print('\nfinished %s'%tag)
    
    return out_dir

def dev():
    """problem with trues?"""
    return run(
        tag='dev',
        catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hyd2_dev\model_run_index.csv',
        modelID_l = None,
        
        compiled_fp_d = {
        'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\dev\20220331\working\outs_analy_dev_0331.pickle',
        'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\dev\20220331\working\agg_mindex_analy_dev_0331.pickle',
        #'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\dev\20220331\working\trues_analy_dev_0331.pickle',
        
 
            }
 
        
        )
    

def r2():
    return run(
        #modelID_l = [0, 11],
        tag='r2',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd2\model_run_index.csv',
        compiled_fp_d = {
              'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220316\working\outs_analy_r2_0316.pickle',
            'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220331\working\agg_mindex_analy_r2_0331.pickle',
            'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220331\working\trues_analy_r2_0331.pickle',
            },
        )

def r3():
    return run(
        #modelID_l = [0, 11],
        tag='r3',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd3\model_run_index.csv',
        compiled_fp_d = {
 
            },
        )

def r4():
    return run(
        #modelID_l = [0, 11],
        tag='r4',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd4\model_run_index.csv',
        compiled_fp_d = {
         'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r4\20220410\working\outs_analy_r4_0410.pickle',
        'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\r4\20220410\working\agg_mindex_analy_r4_0410.pickle',
        #'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\r4\20220410\working\trues_analy_r4_0410.pickle',
            },
        )

def r5():
    return run(
        #modelID_l = [0, 11],
        tag='r5',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd5\model_run_index.csv',
        compiled_fp_d = {
        'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r5\20220412\working\outs_analy_r5_0412.pickle',
        'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\r5\20220412\working\agg_mindex_analy_r5_0412.pickle',
        'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\r5\20220412\working\trues_analy_r5_0412.pickle',
            },
        )
 
if __name__ == "__main__": 
    
 
    #output=dev()
    #output=r2()
    output=r5()
        
        
 
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))