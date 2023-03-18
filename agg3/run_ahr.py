'''
Created on Mar. 18, 2023

@author: cefect

started this to compare aggregated grids to coarse hyd model grids
    but decided to just compute agg grids here... and have a new project for inundation performance
'''
#===============================================================================
# IMPORTS-------
#===============================================================================
import os, copy, datetime, pickle
import numpy as np
import numpy.ma as ma

import rasterio as rio
import shapely.geometry as sgeo
from rasterio.enums import Resampling


from hp.basic import get_dict_str
from hp.hyd import get_wsh_rlay
from hp.rio import (
    get_stats,rlay_apply, get_rlay_shape
    )

from agg3.scripts import Agg3_Session

#===============================================================================
# FUNCS------
#===============================================================================
def ahr_aoi08_r32_0130_30():
    return run_pipeline(
        wse1_fp=r'c:\LS\10_IO\2112_Agg\ins\hyd\ahr\ahr_aoi08_r32_0130_30\ahr_aoi08_r04_1215-0030_wse.tif',
        #wse2_fp=r'c:\LS\10_IO\2112_Agg\ins\hyd\ahr\ahr_aoi08_r32_0130_30\ahr_aoi08_r32_1221-0030_wse.tif',
        dem1_fp=r'c:\LS\10_IO\2112_Agg\ins\hyd\ahr\ahr_aoi08_r32_0130_30\dem005_r04_aoi08_1210.asc',
        aggscale=32/4,
        init_kwargs=dict(run_name='ahr_aoi08_0130'),
        )


def run_pipeline(
        dem1_fp=None,
        wse1_fp=None,       
        aoi_fp=None,
        aggscale=2,
        method_lib={'avgWSH':dict(), 'avgWSE':dict()},
        init_kwargs=dict(),
        ):
    """run the comparison
    
    
    Parameters
    ----------
    
    method_lib: dict
        method key:method_kwargs. see get_agg_byType()
        
        
    """
    
    with Agg3_Session(**init_kwargs) as ses:
        log = ses.logger
        #=======================================================================
        # assemble grids
        #=======================================================================
        fp_d = dict(wse1=wse1_fp, dem1=dem1_fp)
        for k,v in fp_d.items():
            assert os.path.exists(v), k
            
        #start results container
        res_lib = {'hyd':dict(wse1=wse1_fp, dem1=dem1_fp)}
        #=======================================================================
        # clip grids
        #=======================================================================
        if not aoi_fp is None:
            clip_lib = ses.clip_rlays(fp_d, aoi_fp=aoi_fp)
            fp_d = {k:v['clip_fp'] for k,v in clip_lib.items()}
            
        #=======================================================================
        # get hi-res depth
        #=======================================================================
        """needed by get_avgWSH and results"""
        fp_d['wsh1'] = get_wsh_rlay(fp_d['dem1'], fp_d['wse1'], out_dir=ses.out_dir)
        res_lib['hyd']['wsh1'] = fp_d['wsh1']
        #=======================================================================
        # infer the aggregation scale
        #=======================================================================
        #=======================================================================
        # """matching the hyd wse1 and wse2 grids"""
        # support_ratio = ses.get_support_ratio(fp_d['wse1'], fp_d['wse2'])
        # assert support_ratio.is_integer()
        # assert support_ratio>=2
        # aggscale = int(support_ratio)
        # 
        # log.info(f'infered aggscale={aggscale}')
        #=======================================================================
        #=======================================================================
        # build aggregate grids on each method  
        #=======================================================================
        agg_res_lib = ses.get_agg_set(method_lib, aggscale=aggscale,
                        dem_fp=fp_d['dem1'], wse_fp=fp_d['wse1'], wsh_fp=fp_d['wsh1'])
        
        #append to results
        res_lib.update(agg_res_lib)
        
        log.info(f'finished on \n{get_dict_str(res_lib)}')
        
        #=======================================================================
        # write 
        #=======================================================================
        ofp = os.path.join(ses.out_dir, f'{ses.fancy_name}_res.pkl')
        with open(ofp, 'wb') as f:
            pickle.dump(res_lib, f)
            
        log.info(f'wrote to \n    {ofp}')
        
    return ofp
        
if __name__=='__main__':
    ahr_aoi08_r32_0130_30()
        
 