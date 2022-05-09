'''
Created on May 9, 2022

@author: cefect
'''
import os
from qgis.core import QgsCoordinateReferenceSystem
from hp.hyd import HQproj


def convert_wse(
        out_dir=r'C:\LS\10_OUT\2112_Agg\outs\prep\0509'
        ):
    from definitions import proj_lib
    
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    res_lib = dict()
    for studyArea, pars_d in proj_lib.items():
        wse_fp_d = pars_d['wse_fp_d']
        dem_fp = pars_d['dem_fp_d'][10]
        
        crs = QgsCoordinateReferenceSystem('EPSG:%i' % pars_d['EPSG'])
        
        res_lib[studyArea] = dict()
        
        with HQproj(dem_fp=dem_fp, out_dir=out_dir, crs=crs) as ses:
            for lvl, fp in wse_fp_d.items():
                raw_rlay = ses.get_layer(fp, mstore=ses.mstore)
                
                rlay = ses.wse_remove_gw(raw_rlay)
                ses.rlay_write(rlay)
        
        
        
        

if __name__ == "__main__":
    convert_wse()