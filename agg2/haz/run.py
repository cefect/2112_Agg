'''
Created on Aug. 28, 2022

@author: cefect
'''
import os
from definitions import proj_lib

import pandas as pd
idx = pd.IndexSlice


 


 
    
def build_vrt(
        pick_fp = r'C:\LS\10_OUT\2112_Agg\outs\dsmp\SJ_r1\20220828\haz\direct\dsmp_SJ_r1_0828_haz_dsmp.pkl',
        **kwargs):
    
    from agg2.haz.scripts import DownsampleSession as Session
    with Session(**kwargs) as ses:
        ses.run_vrts(pick_fp)
        

def SJ_0829_base(method='direct',
                 fp2=None,
                 ):
    #project data
    proj_name = 'SJ'
    proj_d = proj_lib['SJ']
    
 
    wse_fp=proj_d['wse_fp_d']['hi']
    dem_fp=proj_d['dem_fp_d'][1]
    
    from agg2.haz.scripts import DownsampleSession as Session
    
    #execute
    with Session(proj_name=proj_name, run_name='r4_%s'%method) as ses:
        if fp2 is None:
            fp1 = ses.run_dsmp(dem_fp, wse_fp, method=method, dsc_l=[1, 2**3, 2**6, 2**7, 2**8, 2**9])
            
 
     
             
            fp2 = ses.run_catMasks(fp1)
            
            
            
            vrt_d = ses.run_vrts(fp2)
        
        ses.run_stats(fp2)
        
    return fp2

def SJ_0830_filter(**kwargs):
    return SJ_0829_base(method='filter', **kwargs)


def SJ_0830_direct(**kwargs):
    return SJ_0829_base(method='direct', **kwargs)
        
def open_pick(
        fp =r'C:\LS\10_OUT\2112_Agg\outs\SJ\r3_direct\20220829\haz\stats\SJ_r3_direct_0829_haz_stats.pkl'
        ):
    import pandas as pd
    from hp.pd import view
    df = pd.read_pickle(fp)
 
    view(df)
    
def SJ_plot_matrix_3metric_2methods(
        fp_d  = {
            'direct':r'C:\LS\10_OUT\2112_Agg\outs\SJ\r4_direct\20220830\stats\SJ_r4_direct_0830_stats.pkl',
            'filter':r'C:\LS\10_OUT\2112_Agg\outs\SJ\r4_filter\20220830\stats\SJ_r4_filter_0830_stats.pkl',            
            }        
        ):
    """construct figure from SJ downscale cat results"""
    from agg2.haz.da import DownsampleDASession as Session
    from hp.pd import view
    with Session(proj_name='SJ', run_name='r3_da') as ses:
        #join the simulation results
        dxcol_raw = ses.join_stats(fp_d)
        
        
        """
        view(dxcol_raw)
        """
        #promote pixelLength to index
        map_ser = dxcol_raw.loc[:, idx['direct','all','pixelLength']].rename('pixelLength').astype(int)
        
        dxcol_raw.index = pd.MultiIndex.from_frame(dxcol_raw.index.to_frame().join(map_ser))
        
        #=======================================================================
        # lines: row:metric, col:method, color:dsc
        #=============================================R==========================
        coln_l = ['wd_mean', 'wse_area', 'vol']
        
 
        serx = dxcol_raw.loc[:, idx[:, :, coln_l]].droplevel(0).unstack().reindex(index=coln_l, level=2) 
        
        """
        view(serx)
        """
 
 
        ses.plot_matrix_metric_method_var(serx)
        
        #=======================================================================
        # stackced areas ratios
        #=======================================================================
        
        #slice to just the data we want
        """this is the same for both methods"""
        dxcol = ses.add_frac(dxcol_raw)
        
        dfi = dxcol.loc[:, idx['direct', :, 'frac']].droplevel([0,2], axis=1).droplevel(0, axis=0).drop('all', axis=1)
        
        #ses.plot_dsc_ratios(dfi.dropna())
        
 
 
    
if __name__ == "__main__":
    #pick_fp = SJ_0821()
    
    SJ_0830_filter(fp2=r'C:\LS\10_OUT\2112_Agg\outs\SJ\r3_filter\20220830\haz\cMasks\SJ_r3_filter_0830_haz_cMasks.pkl')
    #SJ_0830_direct(fp2=r'C:\LS\10_OUT\2112_Agg\outs\SJ\r3_direct\20220829\haz\cMasks\SJ_r3_direct_0829_haz_cMasks.pkl')
    #open_pick()
    
    #SJ_plot_matrix_3metric_2methods()
    
    print('finished')
 