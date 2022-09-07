'''
Created on Sep. 6, 2022

@author: cefect
'''
from definitions import proj_lib
from pyproj.crs import CRS



def open_pick(
        fp =r'C:\LS\10_OUT\2112_Agg\outs\SJ\r3_direct\20220829\haz\stats\SJ_r3_direct_0829_haz_stats.pkl'
        ):
    import pandas as pd
    from hp.pd import view
    df = pd.read_pickle(fp)
 
    view(df)



def run(
        pick_fp,
        proj_name='SJ',
        aoi_fp=None,
        ):
    
    proj_d = proj_lib[proj_name]
    
    
    from agg2.expo.scripts import ExpoSession
    with ExpoSession(proj_name=proj_name, run_name='r1_expo', aoi_fp=aoi_fp,
                     crs=CRS.from_user_input(proj_d['EPSG']),
                     ) as wrkr:
        wrkr.build_assetRsc(pick_fp, proj_d['finv_fp'])





if __name__ == "__main__":
    fp=r'C:\LS\10_OUT\2112_Agg\outs\SJ\r3_direct\20220829\haz\cMasks\SJ_r3_direct_0829_haz_cMasks.pkl'
    #open_pick(fp=fp)
    run(fp,
        aoi_fp=r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\aoiT03_0906.geojson')
    
    print('finished')