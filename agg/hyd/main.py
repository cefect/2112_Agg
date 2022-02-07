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
        overwrite=True,
        #=======================================================================
        # #data
        #=======================================================================
        proj_lib =     {
             'obwb':{
                   'EPSG': 2955, 
                  'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\inventory\obwb_2sheds_r1_0106_notShed_cent_aoi06.gpkg', 
                  'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\dem\\obwb_NHC2020_DEM_20210804_5x5_cmp_aoi04.tif', 
                  'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218',
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
        #aggregation
        grid_sizes = [50, 200, 500],
        
        #sampling
        rsamps_method = 'points',
        
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
    
    with Session(tag=tag,proj_lib=proj_lib,overwrite=overwrite,
                 
                 bk_lib = {
                     'rsamps':dict(method=rsamps_method),
                     'finv_gPoly':dict(grid_sizes=grid_sizes),
                     
                     'vid_df':dict(
                            selection_d=selection_d,vid_l = vid_l,vid_sample=vid_sample,max_mod_cnt=max_mod_cnt,
                                    ),
                                          
                     },
                 **kwargs) as ses:
        
        
        ses.plt = plt  #attach local matplotlib init
        
 
        
        #ses.plot_depths()
        
        #summary of total loss
        #ses.write_loss_smry()
        
        #gives a nice 'total model output' chart
        #shows how sensitive the top-line results are to aggregation
        #ses.plot_tloss_bars()
        
        #layers (for making MAPS)
        #=======================================================================
        # ses.write_errs()
        # #ses.get_confusion_matrix()
        #  
        # #shows the spread on total loss values
        # ses.plot_terrs_box(lossType = 'tl', ycoln='delta', ylabel='TL error (gridded - true)')
        # ses.plot_terrs_box(lossType = 'rl', ycoln='delta', ylabel='RL error (gridded - true)')
        #  
        # #shows how errors vary with depth
        #ses.plot_errs_scatter(lossType='rl', ycoln='delta', ylims=None, plot_vf=True)
        ses.plot_errs_scatter()
        # ses.plot_errs_scatter(lossType='tl', ycoln='delta', ylims=None, plot_vf=False)
        #=======================================================================
        
        #ses.plot_accuracy_mat()

        #
        
        out_dir = ses.out_dir
        
    print('finished %s'%tag)
    
    return out_dir
        
    
 
 
def dev():
    return run(
        tag='dev',
        compiled_fp_d = {
    'finv_gPoly':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\finv_gPoly_hyd_dev_0207.pickle',
    'finv_gPoly_id_dxind':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\finv_gPoly_id_dxind_hyd_dev_0207.pickle',
    'finv_agg':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\finv_agg_hyd_dev_0207.pickle',
    'fgdir_dxind':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\fgdir_dxind_hyd_dev_0207.pickle',
    'finv_sg_agg':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\finv_sg_agg_hyd_dev_0207.pickle',
    'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\rsamps_hyd_dev_0207.pickle',
    'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\rloss_hyd_dev_0207.pickle',
    'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\tloss_hyd_dev_0207.pickle',
    #'errs':r'C:\LS\10_OUT\2112_Agg\outs\hyd\dev\20220207\working\errs_hyd_dev_0207.pickle',
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
            #  'SaintJohn': {
            #      'EPSG': 3979, 
            #      'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\finv\\microsoft_0517_aoi13_0116.gpkg',
            #       'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\dem\\HRDEM_0513_r5_filnd_aoi12b.tif',
            #        'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\SaintJohn\\wd\\',
            #        'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\SJ_aoiT01_0119.gpkg',
            #                  }, 
            #  'Calgary': {
            #      'EPSG': 3776, 
            #      'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\finv\\calgary_IBI2016_binvRes_170729_aoi02_0116.gpkg', 
            #      'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\Calgary\\dem\\CoC_WR_DEM_170815_5x5_0126.tif', 
            #      'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\DEV0116',
            #      'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\Calgary\aoi\calgary_aoiT01_0119.gpkg',
            #              }, 
            # 'dP': {
            #     'EPSG': 2950, 
            #     'finv_fp': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\finv\\microsoft_20210506_aoi03_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\dem\\HRDEM_CMM2_0722_fild_aoi03_0116.tif', 
            #     'wd_dir': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\dP\\wd\\',
            #     'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\dP\aoiT01_202220118.gpkg',
            #     },
            #===================================================================
            },
                
        grid_sizes = [
            50, 
            200, 
            #500,
            ],
        
        vid_l = [
            #402, #MURL. straight line. smallest agg error
            798, #Budiyono from super meet 4
            811, #Budiyono from super meet 4
            #794, #budyi.. largest postiive error
            ],

        rsamps_method = 'true_mean',
        trim=True,
        overwrite=True,
        )
        
def r1():
    return run(
        tag='points_r1',
        compiled_fp_d = {
 
    'finv_gPoly':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\finv_gPoly_hyd_points_r1_0205.pickle',
    'finv_gPoly_id_dxind':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\finv_gPoly_id_dxind_hyd_points_r1_0205.pickle',
    'finv_agg':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\finv_agg_hyd_points_r1_0205.pickle',
    'fgdir_dxind':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\fgdir_dxind_hyd_points_r1_0205.pickle',
    'finv_sg_agg':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\finv_sg_agg_hyd_points_r1_0205.pickle',
    'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\rsamps_hyd_points_r1_0205.pickle',
    'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\rloss_hyd_points_r1_0205.pickle',
    'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\tloss_hyd_points_r1_0205.pickle',
    'errs':r'C:\LS\10_OUT\2112_Agg\outs\hyd\points_r1\20220205\working\errs_hyd_points_r1_0205.pickle',
            },
        
        #vid_sample=5,
        vid_l = [
            #402, #MURL. straight line. smallest agg error
            798, #Budiyono from super meet 4
            811, #Budiyono from super meet 4
            #794, #budyi.. largest postiive error (as some RL >100)
            49, #worst case FLEMO
            ],
                
        rsamps_method = 'points',
        trim=False,
        overwrite=True,
        )
    
    
def means_r1():
    return run(
        tag='means_r1',
        compiled_fp_d = {
    'finv_gPoly':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\finv_gPoly_hyd_means_r1_0205.pickle',
    'finv_gPoly_id_dxind':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\finv_gPoly_id_dxind_hyd_means_r1_0205.pickle',
    'finv_agg':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\finv_agg_hyd_means_r1_0205.pickle',
    'fgdir_dxind':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\fgdir_dxind_hyd_means_r1_0205.pickle',
    'finv_sg_agg':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\finv_sg_agg_hyd_means_r1_0205.pickle',
    'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\rsamps_hyd_means_r1_0205.pickle',
    'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\rloss_hyd_means_r1_0205.pickle',
    'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\tloss_hyd_means_r1_0205.pickle',
    'errs':r'C:\LS\10_OUT\2112_Agg\outs\hyd\means_r1\20220205\working\errs_hyd_means_r1_0205.pickle',
            },
        
        #vid_sample=5,
        vid_l = [
            #402, #MURL. straight line. smallest agg error
            798, #Budiyono from super meet 4
            811, #Budiyono from super meet 4
            #794, #budyi.. largest postiive error (as some RL >100)
            49, #worst case FLEMO
            ],
        
        rsamps_method = 'true_mean',
        trim=False,
        )
    
    
if __name__ == "__main__": 
    
    #output=means_r1()
    #output=r1()
    output=dev()
 
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))