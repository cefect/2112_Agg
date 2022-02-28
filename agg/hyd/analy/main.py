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


from scripts import ModelAnalysis

 

def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r2',
        overwrite=True,
 
        #=======================================================================
        # #data
        #=======================================================================
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd2\model_run_index.csv',
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
        
        #=======================================================================
        # individual model summaries---------
        #=======================================================================
        mids = [0,2]
        #=======================================================================
        # ses.write_resvlay(dkey='rsamps', modelID_l=mids)
        # for mid in mids:
        #     ses.plot_model_smry(mid)
        #=======================================================================
            
        #=======================================================================
        # exposure calc analysis----------
        #=======================================================================
        
        #=======================================================================
        # true vals generation
        #=======================================================================
        mids = [0, 11]
        #ses.plot_dkey_mat(modelID_l=mids, dkey='tvals', plot_coln='tval_type', plot_rown='studyArea', plot_type='box', add_label=False)
        
        #=======================================================================
        # #total vals per ag method
        #=======================================================================
        dscale_ids = [0,3,6]
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=dscale_ids, dkey='tvals', plot_type='hist', xlims = (0,2))
        # ses.plot_dkey_mat(modelID_l=dscale_ids, dkey='tvals', plot_type='box', xlims = (0,2))
        #=======================================================================
        
        #=======================================================================
        # ses.plot_compare_mat(dkey='tvals', modelID_l=dscale_ids,
        #                      plot_rown='aggLevel', plot_coln='dscale_meth', aggMethod='sum',
        #                      fmt='png')
        #=======================================================================
        
        #=======================================================================
        # #intersection types (using g50)
        #=======================================================================
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=[3,9,10], dkey='rsamps', plot_rown='samp_method', 
        #                   plot_coln='studyArea', plot_type='hist', drop_zeros=True,
        #                   plot_colr='studyArea', bins=30)
        #=======================================================================
        
        #=======================================================================
        # #hazard vs asset resolution
        #=======================================================================
        mids = list(range(9))
        
        #result values
        #=======================================================================
        # for plot_type in ['hist', 'box']:
        #     ses.plot_dkey_mat(modelID_l=mids, dkey='rsamps', 
        #                       plot_rown='aggLevel', 
        #                       plot_coln='resolution', plot_type=plot_type, drop_zeros=True,
        #                       plot_colr='aggLevel', bins=40)
        # 
        # #error plots
        # fmt='png'
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=mids,plot_rown='aggLevel', plot_coln='resolution', fmt=fmt)
        #=======================================================================
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=[0,3,6], plot_rown='aggLevel', plot_coln='studyArea',  fmt=fmt)
        # ses.plot_compare_mat(dkey='rsamps', modelID_l=[0,1,2], plot_rown='resolution', plot_coln='studyArea', plot_colr='studyArea',  fmt=fmt)
        #=======================================================================
        #=======================================================================
        
        #=======================================================================
        # loss calc analysis: unit loss---------
        #=======================================================================
        mids = [0,12,13]
        
        #=======================================================================
        # inputs plots
        #=======================================================================
        mids = list(range(9))
        #rsamps (res vs aggLevel)
        #=======================================================================
        # ses.plot_dkey_mat(modelID_l=mids, dkey='rsamps', plot_rown='aggLevel', 
        #                   plot_coln='resolution', plot_type='hist', drop_zeros=True,
        #                   plot_colr='aggLevel', bins=30, xlims=(0,5))
        #=======================================================================
        
        
        #=======================================================================
        # unit rloss
        #=======================================================================
        #just points on the vfunc
        #ses.plot_vs_mat(modelID_l=mids, fmt='png')
        
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
        ses.plot_compare_mat(dkey='tloss', modelID_l=[0,3,6], plot_rown='aggLevel', plot_coln='studyArea',  fmt=fmt, aggMethod='sum')
        ses.plot_compare_mat(dkey='tloss', modelID_l=[0,1,2], plot_rown='resolution', plot_coln='studyArea', plot_colr='studyArea',  fmt=fmt, aggMethod='sum')

            

        
        out_dir = ses.out_dir
        
    print('\nfinished %s'%tag)
    
    return out_dir

def dev():
    return run(
        tag='dev',
        catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hyd2_dev\model_run_index.csv',
        modelID_l = None,
        
        compiled_fp_d = {
        'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\dev\20220227\working\outs_analy_dev_0227.pickle',
        
 
            }
 
        
        )
    

def r2():
    return run(
        #modelID_l = [0, 11],
        tag='r2',
        compiled_fp_d = {
  'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220227\working\outs_analy_r2_0227.pickle',
          'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220227\working\agg_mindex_analy_r2_0227.pickle',
        'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220227\working\trues_analy_r2_0227.pickle',
            },
        )
    
if __name__ == "__main__": 
    
 
    #output=dev()
    output=r2()
        
        
 
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))