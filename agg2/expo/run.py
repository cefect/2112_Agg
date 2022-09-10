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



def SJ_r6_0910(
        method='direct',
        fp_lib = {
                'direct':{
                    'catMasks': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\cMasks\\SJ_r6_direct_0909_cMasks.pkl',
                    #'err': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r6\\SJ\\direct\\20220909\\errs\\SJ_r6_direct_0909_errs.pkl',
                    },
                'filter':{
                    'catMasks':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
                    #'err':'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\errs\\SJ_r5_filter_0909_errs.pkl'
                    }
                },
        **kwargs):
    return run_expo(fp_d=fp_lib[method], case_name = 'SJ', method=method)

if __name__ == "__main__":
 
    SJ_r6_0910( 
        aoi_fp=r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\aoiT03_0906.geojson',
        )
    
    print('finished')