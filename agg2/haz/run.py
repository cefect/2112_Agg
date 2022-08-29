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
    proj_name = 'dsmp'
    proj_d = proj_lib['SJ']
    wse_fp=proj_d['wse_fp_d']['hi']
    dem_fp=proj_d['dem_fp_d'][1]
    
    #execute
    with Session(proj_name=proj_name, run_name='SJ_r1') as ses:
        #ofp = ses.run_dsmp(dem_fp, wse_fp, dscList_kwargs=dict(reso_iters=4),method='filter')
        
        dsmp_fp = r'C:\LS\10_OUT\2112_Agg\outs\dsmp\SJ_r1\20220828\haz\filter\dsmp_SJ_r1_0828_haz_dsmp.pkl'
 
        #vrt_d = ses.run_vrts(dsmp_fp)
        
        ses.run_catMasks(dsmp_fp)
        
        
 
    
if __name__ == "__main__":
    #pick_fp = SJ_0821()
    
    SJ_0829()
 