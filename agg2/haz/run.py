'''
Created on Aug. 28, 2022

@author: cefect
'''
import os
from definitions import proj_lib
from agg2.haz.scripts import DownsampleSession as Session


def runr(
        dem_fp=None, wse_fp=None,
        dscList_kwargs = dict(reso_iters=5),
        method='direct',
        **kwargs):
    
    with Session(**kwargs) as ses:
        ofp = ses.run_dsmp(dem_fp, wse_fp, dscList_kwargs=dscList_kwargs,method=method
                           )
        
    return ofp


def SJ_0821():
    proj_name = 'dsmp'
    proj_d = proj_lib['SJ']
    return runr(
        method='filter',dscList_kwargs = dict(reso_iters=4),
        proj_name=proj_name, 
        run_name='SJ_r1', 
        wse_fp=proj_d['wse_fp_d']['hi'],        
        dem_fp=proj_d['dem_fp_d'][1]
        )
    
def build_vrt(
        pick_fp = r'C:\LS\10_OUT\2112_Agg\outs\dsmp\SJ_r1\20220828\haz\direct\dsmp_SJ_r1_0828_haz_dsmp.pkl',
        **kwargs):
    
    with Session(**kwargs) as ses:
        ses.run_vrts(pick_fp)
        

def SJ_0829():
    #project data
    proj_name = 'SJ'
    proj_d = proj_lib['SJ']
    
 
    wse_fp=proj_d['wse_fp_d']['hi']
    dem_fp=proj_d['dem_fp_d'][1]
    
    #execute
    with Session(proj_name=proj_name, run_name='r3_direct') as ses:
        #fp1 = ses.run_dsmp(dem_fp, wse_fp, dscList_kwargs=dict(reso_iters=3),method='direct', dsc_l=[1, 2**3, 2**6, 2**7, 2**8, 2**9])
        
 
        #fp1 = r'C:\LS\10_OUT\2112_Agg\outs\SJ\r2\20220829\haz\dsmp\SJ_r2_0829_haz_dsmp.pkl'
 
         
        #fp2 = ses.run_catMasks(fp1)
        
        fp2=r'C:\LS\10_OUT\2112_Agg\outs\SJ\r3_direct\20220829\haz\cMasks\SJ_r3_direct_0829_haz_cMasks.pkl'
        
        #vrt_d = ses.run_vrts(fp2)
        
        ses.run_stats(fp2)
        
def open_pick(
        fp =r'C:\LS\10_OUT\2112_Agg\outs\SJ\r3_direct\20220829\haz\stats\SJ_r3_direct_0829_haz_stats.pkl'
        ):
    import pandas as pd
    from hp.pd import view
    df = pd.read_pickle(fp)
    
 
    view(df)
 
 
    
if __name__ == "__main__":
    #pick_fp = SJ_0821()
    
    #SJ_0829()
    open_pick()
    
    print('finished')
 