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
matplotlib.rcParams['figure.titlesize'] = 12
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)


from scripts import ModelAnalysis

def write_pars():
    df = get_all_pars()
    df.to_csv(r'C:\LS\10_OUT\2112_Agg\lib\pars.csv', header=True, index=True)
    


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
        modelID_l=[],
 
        #=======================================================================
        # plot control
        #=======================================================================
        transparent=False,
        
        #=======================================================================
        # debugging
        #=======================================================================
        debug_len=None,
        
        **kwargs):
    
    with ModelAnalysis(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                       catalog_fp=catalog_fp,
                       bk_lib = {
                           'outs':dict(modelID_l=modelID_l, debug_len=debug_len)
                           
                           },
                 **kwargs) as ses:
        
        ses.plot_total_bars()
 
        
 
 
        #ses.plot_tvals()
        #ses.plot_depths(calc_str=rsamps_method)
        
        #summary of total loss
        #ses.write_loss_smry()
         
        #gives a nice 'total model output' chart
        #shows how sensitive the top-line results are to aggregation
        #ses.plot_tloss_bars()
        #  
        # #layers (for making MAPS)
        # #ses.write_errs()
        # #ses.get_confusion_matrix()
        # #  
        # # #shows the spread on total loss values
        # ses.plot_terrs_box(ycoln = ('tl', 'delta'), ylabel='TL error (gridded - true)')
        # #ses.plot_terrs_box(ycoln = ('rl', 'delta'), ylabel='RL error (gridded - true)')
        # #
        # #=======================================================================
        # # error scatter plots  
        # #=======================================================================
        # #shows how errors vary with depth
        # ses.plot_errs_scatter(xcoln = ('depth', 'grid'), ycoln = ('rl', 'delta'), xlims = (0, 2), ylims=(-10,100), plot_vf=True)
        # ses.plot_errs_scatter(xcoln = ('depth', 'grid'), ycoln = ('tl', 'delta'), xlims = (0, 2), ylims=None, plot_vf=False)
        #  
        # #vs aggregated counts
        # ses.plot_errs_scatter(xcoln = (Session.scale_cn, 'grid'), ycoln = ('rl', 'delta'), xlims = (0,50), ylims=(-10,100), plot_vf=False)
        # ses.plot_errs_scatter(xcoln = (Session.scale_cn, 'grid'), ycoln = ('tl', 'delta'), xlims = None, ylims=None, plot_vf=False)
        # 
        # """first row on this one doesnt make sense
        # ses.plot_accuracy_mat(plot_zeros=False,lossType = 'tl', binwidth=100, )"""
        # ses.plot_accuracy_mat(plot_zeros=False,lossType = 'rl', binWidth=5)
        # ses.plot_accuracy_mat(plot_zeros=False,lossType = 'depth', binWidth=None,
        #                      lims_d={'raw':{'x':None, 'y':(0,500)}})        
        #=======================================================================
        
        out_dir = ses.out_dir
        
    print('\nfinished %s'%tag)
    
    return out_dir

def dev():
    return run(
        tag='dev',
        catalog_fp=r'C:\LS\10_OUT\2112_Agg\outs\hyd2\dev\20220223\lib\model_run_index.csv',
        modelID_l = [0,1], debug_len=None,
        
        compiled_fp_d = {
'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\dev\20220223\working\outs_analy_dev_0223.pickle',
            }
        
        )
    
def grid_compare():
    return run(
        tag='grid_compare',
        
        modelID_l = [0,1, 2],
        
        compiled_fp_d = {
            'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220223\working\outs_analy_r2_0223.pickle',
            },
        )

if __name__ == "__main__": 
    
    #output=dev()
    output = r2()
        
        
 
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))