'''
Created on Sep. 6, 2022

@author: cefect
'''
from definitions import proj_lib
from pyproj.crs import CRS

 


def run_expo(
        fp_d={},
        case_name = 'SJ',
 
        proj_d=None,
        **kwargs):
    
    #===========================================================================
    # extract parametesr
    #===========================================================================
    #project data   
    if proj_d is None: 
        proj_d = proj_lib[case_name] 
 
    crs = CRS.from_epsg(proj_d['EPSG'])
    finv_fp = proj_d['finv_fp']
    
    from agg2.expo.scripts import ExpoSession
    with ExpoSession(case_name=case_name,crs=crs, nodata=-9999, **kwargs) as ses:
        if not 'arsc' in fp_d:
            fp_d['arsc'] = ses.build_assetRsc(fp_d['catMasks'], finv_fp)
            
        #=======================================================================
        # for layName in [
        #     'wd', 
        #     'wse']:
        #     if not layName in fp_d:
        #         fp_d[layName] = ses.build_layerSamps(fp_d['catMasks'], finv_fp,  layName=layName,write=True,)
        #=======================================================================
            
 
            
        



def SJ_r6_0910(
        method='filter',
        fp_lib = {
                'direct':{
                     'catMasks': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r7\\SJ\\direct\\20220910\\cMasks\\SJ_r7_direct_0910_cMasks.pkl',
                    #'arsc':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\direct\20220910\arsc\SJ_r1_direct_0910_arsc.pkl',
                    'wd':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\direct\20220911\lsamp_wd\SJ_r7_direct_0911_lsamp_wd.pkl',
                    'wse':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\direct\20220911\lsamp_wse\SJ_r7_direct_0911_lsamp_wse.pkl',
                    },
                'filter':{
                    'catMasks':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
                    'arsc':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r1\SJ\filter\20220910\arsc\SJ_r1_filter_0910_arsc.pkl',
                    'wd':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wd\SJ_r7_filter_0911_lsamp_wd.pkl',
                    'wse':r' C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wse\SJ_r7_filter_0911_lsamp_wse.pkl'
                    }
                },
        **kwargs):
    return run_expo(fp_d=fp_lib[method], case_name = 'SJ', method=method,run_name='r8', **kwargs)

if __name__ == "__main__":
 
    SJ_r6_0910(method='direct',
        #aoi_fp=r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\aoiT03_0906.geojson',
        )
    
    print('finished')