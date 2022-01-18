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
        tag='r0',
        proj_lib =     {
            'obwb':{
                  'EPSG': 2955, 
                 'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\inventory\\obwb_2sheds_r1_0106_notShed_cent.gpkg', 
                 'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\dem\\obwb_NHC2020_DEM_20210804_5x5_cmp_aoi04.tif', 
                 'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\wsl\\depth_sB_1218',
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                    }, 
            #===================================================================
            # 'LMFRA': {
            #     'EPSG': 3005, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\finv\\LMFRA_tagComb0612_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\dem\\LMFRA_NHC2019_dtm_5x5_aoi08.tif', 
            #     'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\wd\\0116\\'
            #         }, 
            # 'SaintJohn': {
            #     'EPSG': 3979, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\finv\\microsoft_0517_aoi13_0116.gpkg',
            #      'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\dem\\HRDEM_0513_r5_filnd_aoi12b.tif',
            #       'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\wd\\'
            #                 }, 
            # 'Calgary': {
            #     'EPSG': 3776, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\finv\\calgary_IBI2016_binvRes_170729_aoi02_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\dem\\CoC_WR_DEM_170815_5x5_0126.tif', 
            #     'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\wd\\0116\\'
            #             }, 
            'dP': {
                'EPSG': 2950, 
                'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\finv\\microsoft_20210506_aoi03_0116.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\dem\\HRDEM_CMM2_0722_fild_aoi03_0116.tif', 
                'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\wd\\',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\dP\aoiT01_202220118.gpkg',
                },
            },
        
        **kwargs):
    
    with Session(tag=tag,
                 
                 bk_lib = {
                      
                     'finvg':{'proj_lib':proj_lib},
                     },
                 **kwargs) as ses:
        
        ses.retrieve('rsamps')
 
 
def dev():
    return run(
        compiled_fp_d = {
            'finvg':r'C:\LS\10_OUT\2112_Agg\outs\hyd\r0\20220118\working\finvg_hyd_r0_0118.pickle',
            },
        
        overwrite=True,
        )
        
    
    
    
if __name__ == "__main__": 
    
 
    output=dev()
 
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))