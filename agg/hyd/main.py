'''
Created on Jan. 16, 2022

@author: cefect

explore errors in impact estimates as a result of aggregation using hyd model depths
    let's use hp.coms, but not Canflood
    using damage function csvs from figurero2018 (which were pulled from a db)
    intermediate results only available at Session level (combine Study Areas)
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
import qgis.core
import pandas as pd

#===============================================================================
# import scipy.stats 
# import scipy.integrate
# print('loaded scipy: %s'%scipy.__version__)
#===============================================================================

start = datetime.datetime.now()
print('start at %s' % start)


 


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

#===============================================================================
# custom imports--------
#===============================================================================
from agg.hyd.scripts import Session
#===============================================================================
# FUNCTIONS-------
#===============================================================================
def get_pars_from_xls(
        fp = r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\agg - calcs.xlsx',
        sheet_name='hyd.smry',
        ):
    
    df_raw = pd.read_excel(fp, sheet_name=sheet_name, index_col=0)
    
    d = df_raw.to_dict(orient='index')
    
    print('finished on %i \n    %s\n'%(len(d), d))
    
    

                   
                   

    


def run(
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r0',
        
        #=======================================================================
        # #data
        #=======================================================================
        proj_lib =     {
             'obwb':{
                   'EPSG': 2955, 
                  'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\inventory\obwb_2sheds_r1_0106_notShed_cent_aoi06.gpkg', 
                  'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\dem\\obwb_NHC2020_DEM_20210804_5x5_cmp_aoi04.tif', 
                  'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\DEVdepth_sB_1218',
                  'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                     }, 
            'LMFRA': {
                'EPSG': 3005, 
                'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\finv\\LMFRA_tagComb0612_0116.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\dem\\LMFRA_NHC2019_dtm_5x5_aoi08.tif', 
                'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\DEV0116',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\LMFRA_aoiT01_0119.gpkg',
                    }, 
             'SaintJohn': {
                 'EPSG': 3979, 
                 'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\finv\\microsoft_0517_aoi13_0116.gpkg',
                  'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\dem\\HRDEM_0513_r5_filnd_aoi12b.tif',
                   'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\wd\\',
                   'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\SJ_aoiT01_0119.gpkg',
                             }, 
             'Calgary': {
                 'EPSG': 3776, 
                 'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\finv\\calgary_IBI2016_binvRes_170729_aoi02_0116.gpkg', 
                 'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\dem\\CoC_WR_DEM_170815_5x5_0126.tif', 
                 'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\DEV0116',
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\Calgary\aoi\calgary_aoiT01_0119.gpkg',
                         }, 
            'dP': {
                'EPSG': 2950, 
                'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\finv\\microsoft_20210506_aoi03_0116.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\dem\\HRDEM_CMM2_0722_fild_aoi03_0116.tif', 
                'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\wd\\',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\dP\aoiT01_202220118.gpkg',
                },
            },
        
        #=======================================================================
        # #parameters
        #=======================================================================
        grid_sizes = [50, 200, 500],
        
        #parameters.vfunc selection
        selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              3, #flemo 
                              4, 6, 
                              7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
        
        #parameters.vfunc selection.dev
        vid_l=None,
        vid_sample=None,
        max_mod_cnt=None,
        
        **kwargs):
    
    with Session(tag=tag,proj_lib=proj_lib,
                 
                 bk_lib = {
                     'finvg':dict(grid_sizes=grid_sizes),
                     
                     'vid_df':dict(
                            selection_d=selection_d,vid_l = vid_l,vid_sample=vid_sample,max_mod_cnt=max_mod_cnt,
                                    ),
                                          
                     },
                 **kwargs) as ses:
        
        #attach local matplotlib init
        ses.plt = plt 
        
        #ses.plot_depths()
        ses.plot_tloss_bars()
        
        out_dir = ses.out_dir
        
    print('finished %s'%tag)
    
    return out_dir
        
    
 
 
def dev():
    return run(
        tag='dev',
        compiled_fp_d = {
    #===========================================================================
    # 'finvg':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220119\working\finvg_hyd_dev_0119.pickle',
    # 'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220119\working\rsamps_hyd_dev_0119.pickle',
    # 'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220119\working\rloss_hyd_dev_0119.pickle',
    #===========================================================================
            },
        
                proj_lib =     {
             'obwb':{
                   'EPSG': 2955, 
                  'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\inventory\obwb_2sheds_r1_0106_notShed_cent_aoi06.gpkg', 
                  'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\dem\\obwb_NHC2020_DEM_20210804_5x5_cmp_aoi04.tif', 
                  'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\DEVdepth_sB_1218',
                  'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                     }, 
            'LMFRA': {
                'EPSG': 3005, 
                'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\finv\\LMFRA_tagComb0612_0116.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\dem\\LMFRA_NHC2019_dtm_5x5_aoi08.tif', 
                'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\DEV0116',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\LMFRA_aoiT01_0119.gpkg',
                    }, 
             'SaintJohn': {
                 'EPSG': 3979, 
                 'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\finv\\microsoft_0517_aoi13_0116.gpkg',
                  'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\dem\\HRDEM_0513_r5_filnd_aoi12b.tif',
                   'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\wd\\',
                   'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\SJ_aoiT01_0119.gpkg',
                             }, 
             'Calgary': {
                 'EPSG': 3776, 
                 'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\finv\\calgary_IBI2016_binvRes_170729_aoi02_0116.gpkg', 
                 'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\dem\\CoC_WR_DEM_170815_5x5_0126.tif', 
                 'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\DEV0116',
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\Calgary\aoi\calgary_aoiT01_0119.gpkg',
                         }, 
            'dP': {
                'EPSG': 2950, 
                'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\finv\\microsoft_20210506_aoi03_0116.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\dem\\HRDEM_CMM2_0722_fild_aoi03_0116.tif', 
                'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\wd\\',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\dP\aoiT01_202220118.gpkg',
                },
            },
                
        grid_sizes = [20, 100, 200],
        
        vid_l = [
            402, #MURL. straight line. smallest agg error
            798, #Budiyono from super meet 4
            811, #Budiyono from super meet 4
            794, #budyi.. largest postiive error
            ],

        
        trim=True,
        overwrite=True,
        )
        
def r1():
    return run(
        tag='r1',
        compiled_fp_d = {
 
            },
        
        #vid_sample=5,
        vid_l = [
            402, #MURL. straight line. smallest agg error
            798, #Budiyono from super meet 4
            811, #Budiyono from super meet 4
            #794, #budyi.. largest postiive error
            ],
                
                
        trim=False,
        overwrite=True,
        )
    
def r2():
    return run(
        tag='r2',
        compiled_fp_d = {
 
            },
        
        #vid_sample=5,
        vid_l = [
            402, #MURL. straight line. smallest agg error
            798, #Budiyono from super meet 4
            811, #Budiyono from super meet 4
            #794, #budyi.. largest postiive error
            ],
                
                
        trim=False,
        overwrite=True,
        )
    
    
if __name__ == "__main__": 
    
    #output=r2()
    output=dev()
 
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))