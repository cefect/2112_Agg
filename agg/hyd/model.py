'''
Created on Jan. 16, 2022

@author: cefect

running hyd models (from python IDE)

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
             pars_fp = r'C:\LS\10_OUT\2112_Agg\ins\hyd\model_pars\hyd_modelPars_0226.xls',
             ):
    
    #===========================================================================
    # load pars file
    #===========================================================================
    from numpy import dtype
    #pars_df_raw.dtypes.to_dict()
    #pars_df_raw = pd.read_csv(pars_fp, index_col=False, comment='#')
    pars_df_raw= pd.read_excel(pars_fp, comment='#')
    
    
    #remove notes comumns
    bxcol = pars_df_raw.columns.str.startswith('~')
    if bxcol.any():
        print('dropping %i/%i columns flagged as notes'%(bxcol.sum(), len(bxcol)))
        
        
    pars_df1 = pars_df_raw.loc[:, ~bxcol].dropna(how='all', axis=1).dropna(subset=['modelID'], how='any').infer_objects()
    
    
    pars_df2 = pars_df1.astype(
        {'modelID': int, 'tag': str, 'tval_type': str, 
         'aggLevel': int, 'aggType': str, 'dscale_meth': dtype('O'), 'severity': dtype('O'), 
         'resolution': int, 'downSampling': dtype('O'), 'sgType': dtype('O'), 
         'samp_method': dtype('O'), 'zonal_stat': dtype('O'), 'vid': int}        
        ).set_index('modelID')
    
    #===========================================================================
    # check
    #===========================================================================
    pars_df = pars_df2.copy()
    
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
    raw_d = pars_df.loc[modelID, :].to_dict()
    
    #renamp types
    """
    this is an artifact of loading parameters from pandas
    not very nice.. but not sure how else to preserve the type checks"""
    for k,v in copy.copy(raw_d).items():
        if isinstance(v, str):
            continue
        elif 'int' in type(v).__name__:
            raw_d[k] = int(v)
            
        elif 'float' in type(v).__name__:
            raw_d[k] = float(v)
        
    
    return raw_d
     
    
def run_autoPars( #run a pre-defined model configuration
        modelID=0,
        **kwargs):
    print('START on %i w/ %s'%(modelID, kwargs))
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
        iters=3, trim=True, name='hyd4_dev',**kwargs):
    
    return run_autoPars(iters=iters, trim=trim, name=name, **kwargs)
    
 

def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r2_base',
        name='hyd4',
        overwrite=True,
        trim=False,
        write=True,
        exit_summary=True,
        #=======================================================================
        # #data
        #=======================================================================
        studyArea_l = None, #convenience filtering of proj_lib
        proj_lib =     {
            'obwb':{
                  'EPSG': 2955, 
                 'finv_fp': r'C:\LS\10_OUT\2112_Agg\ins\hyd\obwb\inventory\obwb_2sheds_r1_0106_notShed_aoi06.gpkg', 
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
        dscale_meth='centroid', #downscaling to the aggreated finv
        
        #sampling (geo). see Model.build_sampGeo()
        sgType = 'poly', 
        
        #raster downSampling and selection  (StudyArea.get_raster())
        dsampStage='none', resolution=5, downSampling='none', 
        severity = 'hi', 
        
        #sampling (method). see Model.build_rsamps()
        samp_method = 'zonal', zonal_stat='Mean',  # stats to use for zonal. 2=mean
      
 
        #vfunc selection
        vid = 798, 
        
        #=======================================================================
        # meta
        #=======================================================================
        cat_d={},
        
 
        **kwargs):
    print('START run w/ %s.%s and iters=%i'%(name, tag, iters))
    #===========================================================================
    # parameter logic override
    #===========================================================================
    """these overrides are an artifact of having overly flexible parameters"""
    if tval_type=='uniform':
        iters=1
        
    #===========================================================================
    # study area filtering
    #===========================================================================
    if not studyArea_l is None:
        print('filtering studyarea to %i: %s'%(len(studyArea_l), studyArea_l))
        miss_l = set(studyArea_l).difference(proj_lib.keys())
        assert len(miss_l)==0, 'passed %i studyAreas not in proj_lib: %s'%(len(miss_l), miss_l)
        proj_lib = {k:v for k,v in proj_lib.items() if k in studyArea_l}
    #===========================================================================
    # execute
    #===========================================================================
    with ModelStoch(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                    iters=iters,write=write,exit_summary=exit_summary,
                 bk_lib = {
                     'finv_agg_d':dict(aggLevel=aggLevel, aggType=aggType),
                     
                     
                     'drlay_d':dict( severity=severity, resolution=resolution, downSampling=downSampling, dsampStage=dsampStage),

                     'rsamps':dict(samp_method=samp_method, zonal_stat=zonal_stat,
                                   ),
                     
                     'finv_sg_d':dict(sgType=sgType),
                     
                     'tvals_raw':dict(normed=normed),
                     'tvals':dict(tval_type=tval_type, dscale_meth=dscale_meth),
                     'rloss':dict(vid=vid),
                                          
                     },
                 **kwargs) as ses:
        
        #special library override for dev runs
        if tag=='dev':
            lib_dir = os.path.join(ses.out_dir, 'lib')
            if not os.path.exists(lib_dir):os.makedirs(lib_dir)
        else:
            lib_dir = None
        
        #execute parts in sequence
        ses.run_dataGeneration()
        ses.run_intersection()
        ses.run_lossCalcs()
        
        #write results
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
        
 
        iters=3,
        aggType='gridded', aggLevel=50,
        
        dsampStage='wsl',
        resolution=100, downSampling='Average',
 
        trim=True,
        overwrite=True,
 
        )
    
    
    
 
 
if __name__ == "__main__": 
 
 
 #==============================================================================
 #    output=run_auto_dev(modelID=25, write=True,
 #                        compiled_fp_d={ 
 #  
 # 
 # 
 #                            },
 #                        studyArea_l = ['Calgary'],
 #                        )
 #==============================================================================
    #output=dev()
 
    output=run_autoPars(modelID=25, write=True, name='hyd4_dev', studyArea_l = ['obwb'],
                        compiled_fp_d={
 
                            })
    #output=run_autoPars(tag='g100', modelID=3)
    #output=run_autoPars(tag='g100_true', modelID=4, trim=True)
    #output=run_autoPars(tag='dev', modelID=0, trim=True, iters=3)
    
 
        
 
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))