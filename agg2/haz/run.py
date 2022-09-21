'''
Created on Aug. 28, 2022

@author: cefect
'''
import os, pathlib, pprint
from definitions import proj_lib
from hp.basic import get_dict_str, now
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
        # build aggregated layers
        #=======================================================================
        """todo: add something similar for the layers"""
        if not 'agg' in fp_d:
            fp_d['agg'] = ses.run_agg(dem_fp, wse_fp, method=method, dsc_l=dsc_l)
            ses._clear()
 
        #=======================================================================
        # build difference grids
        #=======================================================================
        if not 'diffs' in fp_d:
            fp_d['diffs'] = ses.run_diffs(fp_d['agg'])
            ses._clear()
            
        #=======================================================================
        # category masks
        #=======================================================================
        if not 'catMasks' in fp_d:
            fp_d['catMasks'] = ses.run_catMasks(fp_d['agg'])            
            ses._clear()
            
        
        #=======================================================================
        # assemble vrts
        #=======================================================================
        for k, v in fp_d.items(): 
            vrt_d = ses.run_vrts(v, layname=k, logger=log.getChild(k))  
 
            
        #=======================================================================
        # build agg stats
        #=======================================================================

        stat_d['s2'] = ses.run_stats(fp_d['agg'], fp_d['catMasks'])
        stat_d['s1'] = ses.run_stats_fine(fp_d['agg'], fp_d['catMasks'])
        
        #=======================================================================
        # build difference stats
        #=======================================================================
        if 'diffs' in fp_d:
            stat_d['diffs'] =ses.run_diff_stats(fp_d['diffs'], fp_d['catMasks'])
        
        ses.concat_stats(stat_d)

        #=======================================================================
        # wrap
        #=======================================================================
        
        log.info('finished w/ array picks \n%s'%pprint.pformat(fp_d, width=10, indent=3, compact=True, sort_dicts =False))
        log.info('finished w/ stat picks \n%s'%pprint.pformat(stat_d, width=30, indent=3, compact=True, sort_dicts =False))
        out_dir = ses.out_dir
        
        
    print('finished to %s'%out_dir)

    return fp_d, stat_d

 
def build_vrt(pick_fp = None,**kwargs):
    
    from agg2.haz.scripts import UpsampleSession as Session
    with Session(**kwargs) as ses:
        ses.run_vrts(pick_fp)
        

#===============================================================================
# def SJ_r5_0909(run_name='r8',
#         method='direct',
#         fp_lib = {
#                 'direct':{
#                     'catMasks': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\SJ\direct\20220917\cMasks\SJ_r8_direct_0917_cMasks.pkl',
#                     #'err': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\errs\\SJ_r6_direct_0909_errs.pkl',
#                     },
#                 'filter':{
#                     'catMasks':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
#                     #'err':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\errs\\SJ_r5_filter_0909_errs.pkl'
#                     }
#                 },
#         **kwargs):
#     return run_haz_agg2(case_name='SJ', fp_d = fp_lib[method], method=method, run_name=run_name, **kwargs)
#===============================================================================
 
    
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

        

def SJ_r9_0921(run_name='r9',
        method='direct',
        fp_lib = {
                'direct':{
                    #'catMasks': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\SJ\direct\20220917\cMasks\SJ_r8_direct_0917_cMasks.pkl',
                    #'err': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\errs\\SJ_r6_direct_0909_errs.pkl',
                    },
                'filter':{
                    #'catMasks':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
                    #'err':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\errs\\SJ_r5_filter_0909_errs.pkl'
                    }
                },
        **kwargs):
    return run_haz_agg2(case_name='SJ', fp_d = fp_lib[method], method=method, run_name=run_name, **kwargs)


if __name__ == "__main__": 
    start = now()
 
    
    SJ_r9_0921(method='direct',
                #dsc_l=[1,  2**7],
                #run_name='t'
               )
 
 
    print('finished in %.2f'%((now()-start).total_seconds())) 