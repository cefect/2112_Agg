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
import os, datetime, math, pickle, copy, sys
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
from agg.hyd.scripts import Model, ModelStoch, get_all_pars, view, Error

#===========================================================================
# #presets
#===========================================================================
 



    
#===============================================================================
# FUNCTIONS-------
#===============================================================================
def get_pars(#retrieving and pre-checking parmeter values based on model ID
            modelID,
            #file with preconfigrued runs
             pars_fp = r'C:\LS\10_OUT\2112_Agg\ins\hyd\model_pars\hyd_modelPars_0226.csv',
             ):
    
    #===========================================================================
    # load pars file
    #===========================================================================
    from numpy import dtype
    #pars_df_raw.dtypes.to_dict()
    pars_df_raw = pd.read_csv(pars_fp, index_col=False, comment='#')
    pars_df = pars_df_raw.dropna(how='all').infer_objects().astype(
        {'modelID': int, 'tag': str, 'tval_type': str, 
         'aggLevel': int, 'aggType': str, 'dscale_meth': dtype('O'), 'severity': dtype('O'), 
         'resolution': int, 'resampling': dtype('O'), 'sgType': dtype('O'), 
         'samp_method': dtype('O'), 'zonal_stat': dtype('O'), 'vid': int}        
        ).set_index('modelID')
    
    assert pars_df.notna().all().all()
    assert pars_df.index.is_unique
    assert pars_df['tag'].is_unique
    assert pars_df.index.name == 'modelID'
 
    assert modelID in pars_df.index
    
    #===========================================================================
    # check
    #===========================================================================
    #possible paramater combinations
    pars_lib = copy.deepcopy(Model.pars_lib)
 
    
    #value check
    for id, row in pars_df.iterrows():
        """replacing nulls so these are matched'"""
        #bx = pars_df.fillna('nan').eq(pre_df.fillna('nan').loc[id, :])
    
        """
        view(pars_df.join(bx.sum(axis=1).rename('match_cnt')).sort_values('match_cnt', ascending=False))
        """
        
        #check each parameter value
        for varnm, val in row.items():
            
            #skippers
            if varnm in ['tag']:
                continue 
 
            allowed_l = pars_lib[varnm]['vals']
            
            if not val in allowed_l:
                
                raise Error(' modelID=%i \'%s\'=\'%s\' not in allowed set\n    %s'%(
                id, varnm, val, allowed_l))
        
    #type check
    for varnm, dtype in pars_df.dtypes.items():
        if varnm in ['tag']:continue
        v1 = pars_lib[varnm]['vals'][0]
        
        if isinstance(v1, str):
            assert dtype.char=='O'
        elif isinstance(v1, int):
            assert 'int' in dtype.name
        elif isinstance(v1, float):
            assert 'float' in dtype.name
 
                
 
            
        
 
    
    #===========================================================================
    # get kwargs
    #===========================================================================
    
    return pars_df.loc[modelID, :].to_dict()
     
    
def run_autoPars( #retrieve pars from container
        modelID=0,
        **kwargs):
    
    #retrieve preconfigured parameters
    model_pars = get_pars(modelID)
    
    #reconcile passed parameters
    for k,v in copy.copy(model_pars).items():
        if k in kwargs:
            if not v==kwargs[k]:
                print('WARNING!! passed parameter \'%s\' conflicts with pre-loaded value...replacing'%(k))
                model_pars[k] = kwargs[k] #overwrite these for reporting
 
        
    
    return run(
        modelID=modelID,
        cat_d=copy.deepcopy(model_pars),
        **{**model_pars, **kwargs} #overwrites model_pars w/ kwargs (where theres a conflict)
        )
    
def run_auto_dev( #special dev runner
        iters=10, trim=True, name='hyd2_dev',**kwargs):
    
    return run_autoPars(iters=iters, trim=trim, name=name, **kwargs)
    
 

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
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218_fnd\depth_sB_0500_1218fnd.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218_fnd\depth_sB_0100_1218fnd.tif',
                      },
                     }, 
            'LMFRA': {
                'EPSG': 3005, 
                'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\finv\IBI_BldgFt_V3_20191213_aoi08_0219.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\dem\\LMFRA_NHC2019_dtm_5x5_aoi08.tif', 
                #'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\DEV0116',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\LMFRA_aoiT01_0119.gpkg',
                'wd_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116_fnd\AG4_Fr_0500_dep_0116_cmpfnd.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116_fnd\AG4_Fr_0100_dep_0116_cmpfnd.tif',
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
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\0116_fnd\IBI_2017CoC_s0_0500_170729_dep_0116fnd.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\0116_fnd\IBI_2017CoC_s0_0100_170729_dep_0116fnd.tif',
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
        aggType = 'none', aggLevel = 0,
        
        #down scaling (asset values)
        tval_type = 'rand', normed=True, #generating true asset values
        dscale_meth='centroid_inter', #downscaling to the aggreated finv
        
        #sampling (geo). see Model.build_sampGeo()
        sgType = 'poly', 
        
        #raster resampling and selection  (StudyArea.get_raster())
        resolution=0, resampling='none', severity = 'hi',
        
        #sampling (method). see Model.build_rsamps()
        samp_method = 'zonal', zonal_stat='Mean',  # stats to use for zonal. 2=mean
      
 
        #vfunc selection
        vid = 798, 
        
        #=======================================================================
        # meta
        #=======================================================================
        cat_d={},
        
 
        **kwargs):
    
    #===========================================================================
    # update depth rastsers
    #===========================================================================
 
 
    #===========================================================================
    # execute
    #===========================================================================
    with ModelStoch(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                    iters=iters,
                 bk_lib = {
                     'finv_agg_d':dict(aggLevel=aggLevel, aggType=aggType),

                     'rsamps':dict(samp_method=samp_method, zonal_stat=zonal_stat, severity=severity,
                                   resolution=resolution, resampling=resampling),
                     
                     'finv_sg_d':dict(sgType=sgType),
                     
                     'tvals_raw':dict(tval_type=tval_type, normed=normed),
                     'tvals':dict(dscale_meth=dscale_meth),
                     'vfunc':dict(vid=vid),
                                          
                     },
                 **kwargs) as ses:
        
        #special library override for dev runs
        if tag=='dev':
            lib_dir = os.path.join(ses.out_dir, 'lib')
            if not os.path.exists(lib_dir):os.makedirs(lib_dir)
        else:
            lib_dir = None
        
        #ses.retrieve('rsamps')
        ses.write_summary()
        ses.write_lib(lib_dir=lib_dir, cat_d=cat_d)
 
        
        out_dir = ses.out_dir
        
    print('\nfinished %s'%tag)
    
    return out_dir


 
 
def dev():
    
    return run(
        tag='dev',modelID = 1,
        compiled_fp_d = {
 
            },
        
        proj_lib =     {
            'obwb':{
                  'EPSG': 2955, 
                 'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\inventory\obwb_2sheds_r1_0106_notShed_aoi06.gpkg', 
                 'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\obwb\\dem\\obwb_NHC2020_DEM_20210804_5x5_cmp_aoi04.tif', 
                 #'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218',
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                  'wd_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218_fnd\depth_sB_0500_1218fnd.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\wsl\depth_sB_1218_fnd\depth_sB_0100_1218fnd.tif',
                      },
                    }, 
            'LMFRA': {
                'EPSG': 3005, 
                'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\finv\IBI_BldgFt_V3_20191213_aoi08_0219.gpkg', 
                'dem': 'C:\\LS\\10_OUT\\2112_Agg\\ins\\hyd\\LMFRA\\dem\\LMFRA_NHC2019_dtm_5x5_aoi08.tif', 
                #'wd_dir': r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\DEV0116',
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\LMFRA_aoiT01_0119.gpkg',
                'wd_fp_d':{
                      'hi':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116_fnd\AG4_Fr_0500_dep_0116_cmpfnd.tif',
                      'low':r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116_fnd\AG4_Fr_0100_dep_0116_cmpfnd.tif',
                      },
                    }, 
            },
        iters=3,
        aggType='gridded', aggLevel=50,
        resolution=100, resampling='Average',
        #=======================================================================
        # #aggType = 'none', aggLevel = None,
        # aggType = 'none', aggLevel = None,
        # tval_type = 'rand', #see build_tvals()
        # 
        # samp_method = 'zonal', sgType='poly', zonal_stat='Mean',
        #=======================================================================
        trim=True,
        overwrite=True,
 
        )
    
    
    
def r2_base():
    
    return run(
        tag='base',  modelID = 0,
       
        compiled_fp_d = {
     'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\base\20220223\working\finv_agg_d\finv_agg_d_hyd2_base_0223.pickle',
    'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\base\20220223\working\finv_agg_mindex_hyd2_base_0223.pickle',
    'tvals':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\base\20220223\working\tvals_hyd2_base_0223.pickle',
    'finv_sg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\base\20220223\working\finv_sg_d\finv_sg_d_hyd2_base_0223.pickle',
    'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\base\20220223\working\rsamps_hyd2_base_0223.pickle',
    'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\base\20220223\working\rloss_hyd2_base_0223.pickle',
    'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\base\20220223\working\tloss_hyd2_base_0223.pickle',
            }
        )
    
 
    
def r2_g200():
    return run(
        tag='g200', modelID=1,
        aggType='gridded', aggLevel=200,
        
        compiled_fp_d = {
    'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\g200\20220223\working\finv_agg_d\finv_agg_d_hyd2_g200_0223.pickle',
    'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\g200\20220223\working\finv_agg_mindex_hyd2_g200_0223.pickle',
    'tvals':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\g200\20220223\working\tvals_hyd2_g200_0223.pickle',
    'finv_sg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\g200\20220223\working\finv_sg_d\finv_sg_d_hyd2_g200_0223.pickle',
    'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\g200\20220223\working\rsamps_hyd2_g200_0223.pickle',
    'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\g200\20220223\working\rloss_hyd2_g200_0223.pickle',
    'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2\g200\20220223\working\tloss_hyd2_g200_0223.pickle',
            
            }
        )
    
def base_dev():

    return run_auto_dev(modelID=0,
        compiled_fp_d = {
              'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2_dev\base_dev\20220226\working\finv_agg_d\finv_agg_d_hyd2_dev_base_dev_0226.pickle',
            'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hyd2_dev\base_dev\20220226\working\finv_agg_mindex_hyd2_dev_base_dev_0226.pickle',
            'tvals':r'C:\LS\10_OUT\2112_Agg\outs\hyd2_dev\base_dev\20220226\working\tvals_hyd2_dev_base_dev_0226.pickle',
            'finv_sg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd2_dev\base_dev\20220226\working\finv_sg_d\finv_sg_d_hyd2_dev_base_dev_0226.pickle',
            'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd2_dev\base_dev\20220226\working\rsamps_hyd2_dev_base_dev_0226.pickle',
            'rloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2_dev\base_dev\20220226\working\rloss_hyd2_dev_base_dev_0226.pickle',
            'tloss':r'C:\LS\10_OUT\2112_Agg\outs\hyd2_dev\base_dev\20220226\working\tloss_hyd2_dev_base_dev_0226.pickle',
        })
 
if __name__ == "__main__": 
    
    #output=base_dev()
    run_auto_dev(modelID=2)
    #output=dev()
    #output=r2_base()
    #output=r2_g200()
    #output=run_autoPars(tag='g50', modelID=2)
    #output=run_autoPars(tag='g100', modelID=3)
    #output=run_autoPars(tag='g100_true', modelID=4, trim=True)
    #output=run_autoPars(tag='dev', modelID=0, trim=True, iters=3)
    
 
        
 
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))