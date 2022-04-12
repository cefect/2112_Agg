'''
Created on Mar. 25, 2022

@author: cefect
'''
import os
proj_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef'

proj_lib =     { #studyArea data for hyd.runr
            'obwb':{
                  'EPSG': 2955, 
                 'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\inventory\obwb_2sheds_r1_0106_notShed_aoi06_0410.gpkg', 
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
               'wse_fp_d':{
                    'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\wse_sB_1223\10\wse_sB_0100_1218_10.tif',  
                     'mid':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\wse_sB_1223\10\wse_sB_0200_1218_10.tif',
                     'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\wse_sB_1223\10\wse_sB_0500_1218_10.tif',
                     },
               'dem_fp_d':{
                    1:r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\dem\obwb_NHC2020_DEM_20210804_01_aoi07_0304.tif',
                    5:r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\dem\obwb_NHC2020_DEM_20210804_05_aoi07_0304b.tif',
                       },
                    }, 
            
            'LMFRA': {
                'EPSG': 3005, 
                'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\finv\IBI_BldgFt_V3_20191213_aoi08_0408.gpkg', 
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\LMFRA_aoiT01_0119.gpkg',
                'wse_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wsl\AG4_Fr_0500_WL_simu_0415_aoi09_0304.tif', #10x10
                      'mid':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wsl\AG4_Fr_0200_WL_simu_0415_aoi09_0304.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wsl\AG4_Fr_0100_WL_simu_0415_aoi09_0304.tif',
                      },
                'dem_fp_d':{
                     1:r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\dem\LMFRA_NHC2019_dtm_01_aoi09_0304.tif',
                     5:r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\dem\LMFRA_NHC2019_dtm_05_aoi09_0304.tif',
                      },
                    }, 
              
            'Calgary': {
                'EPSG': 3776, 
                'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\finv\ISS_bldgs_das2017_20180501_aoi02_0410.gpkg', 
    
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\Calgary\aoi\calgary_aoiT01_0119.gpkg',
   
                'wse_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wse\sc0\IBI_2017CoC_s0_0500_170729_aoi01_0304.tif',
                      'mid':r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wse\sc0\IBI_2017CoC_s0_0200_170729_aoi01_0304.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wse\sc0\IBI_2017CoC_s0_0100_170729_aoi01_0304.tif',
                      },
                'dem_fp_d':{
                      1:r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\dem\CoC_WR_DEM_170815_01_aoi01_0304.tif',
                      5:r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\dem\CoC_WR_DEM_170815_05_aoi01_0304.tif',
    
                      },
                        }, 
                        
            #===================================================================
            # 'SaintJohn': {
            #     'EPSG': 3979, 
            #     'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\SaintJohn\finv\microsoft_0218_aoi13.gpkg',
            #      'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\dem\\HRDEM_0513_r5_filnd_aoi12b.tif',

            #       'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\SJ_aoiT01_0119.gpkg',
            #                 }, 
            #===================================================================

            #===================================================================
            # 'dP': {
            #     'EPSG': 2950, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\finv\\microsoft_20210506_aoi03_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\dem\\HRDEM_CMM2_0722_fild_aoi03_0116.tif', 
 
            #     'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\dP\aoiT01_202220118.gpkg',
            #     },
            #===================================================================
            }

logcfg_file=r'C:\LS\09_REPOS\01_COMMON\coms\logger.conf'

root_dir=r'C:\LS\10_OUT\2112_Agg'