'''
Created on Aug. 28, 2022

@author: cefect
'''
import os, pathlib
from definitions import proj_lib
from hp.basic import get_dict_str
import pandas as pd
idx = pd.IndexSlice



        

def run_haz_agg2(method='direct',
            fp_d={},
            case_name = 'SJ',
            dsc_l=[1, 2**3, 2**5, 2**6, 2**7, 2**8, 2**9],
            proj_d=None,
                 **kwargs):
    """hazard/raster run for agg2"""
    #===========================================================================
    # imports
    #===========================================================================
 
 
    

    from rasterio.crs import CRS
    
    #===========================================================================
    # extract parametesr
    #===========================================================================
    #project data   
    if proj_d is None: 
        proj_d = proj_lib[case_name] 
    wse_fp=proj_d['wse_fp_d']['hi']
    dem_fp=proj_d['dem_fp_d'][1] 
    crs = CRS.from_epsg(proj_d['EPSG'])
    #===========================================================================
    # run model
    #===========================================================================
    from agg2.haz.scripts import UpsampleSession as Session    
    #execute
    with Session(case_name=case_name,method=method,crs=crs, nodata=-9999, **kwargs) as ses:
        stat_d = dict()
        log = ses.logger
        #=======================================================================
        # build category masks
        #=======================================================================
        """todo: add something similar for the layers"""
        if not 'catMasks' in fp_d:
            fp1 = ses.run_agg(dem_fp, wse_fp, method=method, dsc_l=dsc_l)
 
            fp_d['catMasks'] = ses.run_catMasks(fp1)
        

        
        #=======================================================================
        # build difference grids
        #=======================================================================
        #=======================================================================
        # if not 'err' in fp_d:
        #     fp_d['err'] = ses.run_errs(fp_d['catMasks'])
        #=======================================================================
            
        
        #=======================================================================
        # assemble vrts
        #=======================================================================
 
        vrt_d = ses.run_vrts(fp_d['catMasks'])  
 
        
 
            
        #=======================================================================
        # build agg stats
        #=======================================================================

        #=======================================================================
        # stat_d['s2'] = ses.run_stats(fp_d['catMasks'])
        # stat_d['s1'] = ses.run_stats_fine(fp_d['catMasks'])
        #=======================================================================
        
        #=======================================================================
        # build difference stats
        #=======================================================================
        if 'err' in fp_d:
            stat_d['diff'] =ses.run_errStats(fp_d['err'])
        

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished w/ array picks \n%s'%get_dict_str(fp_d))
        log.info('finished w/ stat picks \n%s'%get_dict_str(stat_d))
        out_dir = ses.out_dir
    print('finished to \n    %s'%out_dir)

    return fp_d, stat_d

 
def build_vrt(pick_fp = None,**kwargs):
    
    from agg2.haz.scripts import UpsampleSession as Session
    with Session(**kwargs) as ses:
        ses.run_vrts(pick_fp)
        

def SJ_r5_0909(
        method='direct',
        fp_lib = {
                'direct':{
                    'catMasks': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r7\\SJ\\direct\\20220910\\cMasks\\SJ_r7_direct_0910_cMasks.pkl',
                    #'err': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\errs\\SJ_r6_direct_0909_errs.pkl',
                    },
                'filter':{
                    'catMasks':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
                    #'err':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\errs\\SJ_r5_filter_0909_errs.pkl'
                    }
                },
        **kwargs):
    return run_haz_agg2(case_name='SJ', fp_d = fp_lib[method], method=method, run_name='r7', **kwargs)
 
    
#===============================================================================
# def SJ_r7_0910(
#         method='direct',
#         fp_lib = {
#                 'direct':{
#                     #===========================================================
#                     # 'catMasks': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\cMasks\\SJ_r6_direct_0909_cMasks.pkl',
#                     # 'err': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\errs\\SJ_r6_direct_0909_errs.pkl',
#                     #===========================================================
#                     },
#                 'filter':{
#                     #===========================================================
#                     # 'catMasks':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
#                     # 'err':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\errs\\SJ_r5_filter_0909_errs.pkl'
#                     #===========================================================
#                     }
#                 },
#         **kwargs):
#     return run_haz_agg2(case_name='SJ', fp_d = fp_lib[method], method=method, run_name='r7', **kwargs)
#===============================================================================

        

 
    
if __name__ == "__main__": 
 
    
    SJ_r5_0909(method='direct')
 
 
 
 
    
    
    
    print('finished')
 