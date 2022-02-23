'''
Created on Jan. 16, 2022

@author: cefect

explore errors in impact estimates as a result of aggregation using hyd model depths
    let's use hp.coms, but not Canflood
    using damage function csvs from figurero2018 (which were pulled from a db)
    intermediate results only available at Session level (combine Study Areas)
    
    
key questions
    how do errors relate to depth?
    how much total error might I expect?
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
import qgis.core
import pandas as pd
import numpy as np
np.random.seed(100)
#===============================================================================
# import scipy.stats 
# import scipy.integrate
# print('loaded scipy: %s'%scipy.__version__)
#===============================================================================

start = datetime.datetime.now()
print('start at %s' % start)

 
#===============================================================================
# custom imports--------
#===============================================================================
from agg.hyd.scripts import Model, ModelStoch
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
    
 
    
 

def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r2_base',
        name='hyd2',
        overwrite=True,
        trim=False,
        #=======================================================================
        # #data
        #=======================================================================
        proj_lib =     {
             'obwb':{
                   'EPSG': 2955, 
                  'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\inventory\obwb_2sheds_r1_0106_notShed_aoi06.gpkg', 
                  'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\dem\\obwb_NHC2020_DEM_20210804_5x5_cmp_aoi04.tif', 
                  #'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218',
                  'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                  'wd_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218\depth_sB_0500_1218.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218\depth_sB_0100_1218.tif',
                      },
                     }, 
            'LMFRA': {
                'EPSG': 3005, 
                'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\finv\IBI_BldgFt_V3_20191213_aoi08_0219.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\dem\\LMFRA_NHC2019_dtm_5x5_aoi08.tif', 
                #'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\DEV0116',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\LMFRA_aoiT01_0119.gpkg',
                'wd_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116\AG4_Fr_0500_dep_0116_cmp.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116\AG4_Fr_0100_dep_0116_cmp.tif',
                      },
                    }, 
            #===================================================================
            # 'SaintJohn': {
            #     'EPSG': 3979, 
            #     'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\SaintJohn\finv\microsoft_0218_aoi13.gpkg',
            #      'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\dem\\HRDEM_0513_r5_filnd_aoi12b.tif',
            #     #'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\wd\\',
            #       'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\SJ_aoiT01_0119.gpkg',
            #                 }, 
            #===================================================================
            'Calgary': {
                'EPSG': 3776, 
                'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\finv\ISS_bldgs_das2017_20180501_aoi02.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\dem\\CoC_WR_DEM_170815_5x5_0126.tif', 
                #'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\DEV0116',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\Calgary\aoi\calgary_aoiT01_0119.gpkg',
                 'wd_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\0116\IBI_2017CoC_s0_0500_170729_dep_0116.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\0116\IBI_2017CoC_s0_0100_170729_dep_0116.tif',
                      },
                        }, 
            #===================================================================
            # 'dP': {
            #     'EPSG': 2950, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\finv\\microsoft_20210506_aoi03_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\dem\\HRDEM_CMM2_0722_fild_aoi03_0116.tif', 
            #     'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\wd\\',
            #     'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\dP\aoiT01_202220118.gpkg',
            #     },
            #===================================================================
            },
        
        #=======================================================================
        # #parameters
        #=======================================================================
        #stochasticity
        iters=50,
        
        
        #aggregation
        aggType = 'none', aggLevel = None,
        
        #asset values
        tval_type = 'rand',
        
        #sampling (geo). see Model.build_sampGeo()
        sgType = 'poly', 
        
        #sampling (method). see Model.build_rsamps()
        samp_method = 'zonal', zonal_stat='Mean',  # stats to use for zonal. 2=mean
        severity = 'hi', #hazard raster selection        
        
        
        #vfunc selection
        vid = 798, 
        
 
        **kwargs):
    
    #===========================================================================
    # update depth rastsers
    #===========================================================================
 
    
    if aggType == 'none': assert aggLevel is None
    #===========================================================================
    # execute
    #===========================================================================
    with ModelStoch(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                    iters=iters,
                 bk_lib = {
                     'finv_agg_d':dict(aggLevel=aggLevel, aggType=aggType),

                     'rsamps':dict(samp_method=samp_method, zonal_stat=zonal_stat, severity=severity),
                     
                     'finv_sg_d':dict(sgType=sgType),

                     'tvals':dict(tval_type=tval_type),
                     'vfunc':dict(vid=vid),
                                          
                     },
                 **kwargs) as ses:
        
        #special library override for dev runs
        if tag=='dev':
            lib_dir = os.path.join(ses.out_dir, 'lib')
            if not os.path.exists(lib_dir):os.makedirs(lib_dir)
        else:
            lib_dir = None
        
        ses.write_summary()
        ses.write_lib(lib_dir=lib_dir)
 
        
        out_dir = ses.out_dir
        
    print('\nfinished %s'%tag)
    
    return out_dir
        
    
 
 
def dev():
    
    return run(
        tag='dev',
        compiled_fp_d = {
 
    'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\dev\20220223\working\finv_agg_d\finv_agg_d_hyd2_dev_0223.pickle',
    'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\dev\20220223\working\finv_agg_mindex_hyd2_dev_0223.pickle',
    'tvals':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\dev\20220223\working\tvals_hyd2_dev_0223.pickle',
    'finv_sg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\dev\20220223\working\finv_sg_d\finv_sg_d_hyd2_dev_0223.pickle',
    'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\dev\20220223\working\rsamps_hyd2_dev_0223.pickle',
    'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\dev\20220223\working\rloss_hyd2_dev_0223.pickle',
    'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\dev\20220223\working\tloss_hyd2_dev_0223.pickle',
            },
        
        proj_lib =     {
            'obwb':{
                  'EPSG': 2955, 
                 'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\inventory\obwb_2sheds_r1_0106_notShed_aoi06.gpkg', 
                 'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\dem\\obwb_NHC2020_DEM_20210804_5x5_cmp_aoi04.tif', 
                 #'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218',
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                  'wd_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218\depth_sB_0500_1218.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218\depth_sB_0100_1218.tif',
                      },
                    }, 
            'LMFRA': {
                'EPSG': 3005, 
                'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\finv\IBI_BldgFt_V3_20191213_aoi08_0219.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\dem\\LMFRA_NHC2019_dtm_5x5_aoi08.tif', 
                #'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\DEV0116',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\LMFRA_aoiT01_0119.gpkg',
                'wd_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116\AG4_Fr_0500_dep_0116_cmp.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116\AG4_Fr_0100_dep_0116_cmp.tif',
                      },
                    }, 
            },
        iters=3,
        #aggType = 'none', aggLevel = None,
        aggType = 'none', aggLevel = None,
        tval_type = 'rand', #see build_tvals()
        
        samp_method = 'zonal', sgType='poly', zonal_stat='Mean',
        trim=True,
        overwrite=True,
 
        )
    
def r2_base():
    return run(
        tag='r2_base',
 
        compiled_fp_d = {
                'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_base\20220223\working\finv_agg_d\finv_agg_d_hyd2_r2_base_0223.pickle',
                'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_base\20220223\working\finv_agg_mindex_hyd2_r2_base_0223.pickle',
                'tvals':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_base\20220223\working\tvals_hyd2_r2_base_0223.pickle',
                'finv_sg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_base\20220223\working\finv_sg_d\finv_sg_d_hyd2_r2_base_0223.pickle',
                'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_base\20220223\working\rsamps_hyd2_r2_base_0223.pickle',
                'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_base\20220223\working\rloss_hyd2_r2_base_0223.pickle',
                'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_base\20220223\working\tloss_hyd2_r2_base_0223.pickle',
            }
        )
    
def r2_p01():
    return run(
        tag='r2_p01',
        aggType='gridded', aggLevel=200,
        
        compiled_fp_d = {
            'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_p01\20220223\working\finv_agg_d\finv_agg_d_hyd2_r2_p01_0223.pickle',
            'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_p01\20220223\working\finv_agg_mindex_hyd2_r2_p01_0223.pickle',
            'tvals':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_p01\20220223\working\tvals_hyd2_r2_p01_0223.pickle',
            'finv_sg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_p01\20220223\working\finv_sg_d\finv_sg_d_hyd2_r2_p01_0223.pickle',
            'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_p01\20220223\working\rsamps_hyd2_r2_p01_0223.pickle',
            'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_p01\20220223\working\rloss_hyd2_r2_p01_0223.pickle',
            'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\r2_p01\20220223\working\tloss_hyd2_r2_p01_0223.pickle',
            
            }
        )
        

if __name__ == "__main__": 
    
    #output=dev()
    #output=r2_base()
    output=r2_p01()
        
        
 
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))