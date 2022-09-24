'''
Created on Aug. 28, 2022

@author: cefect
'''
import os, pathlib, pprint, webbrowser
from definitions import proj_lib
from hp.basic import get_dict_str, now, today_str
import shapely.geometry as sgeo
import numpy as np
import pandas as pd
idx = pd.IndexSlice
from dask.distributed import Client
from rasterio.crs import CRS
import xarray as xr
import rioxarray
from dask.diagnostics import ProgressBar 
import dask

from hp.pd import view


res_fp_lib = {'r9':
              {
                'direct':{  
                    'agg': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\agg\\SJ_r9_direct_0921_agg.pkl',
                    'aggXR':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r9\SJ\direct\20220923\aggXR\SJ_r9_direct_0923_aggXR.nc',
                    'diffs': r'C:\LS\10_IO\2112_Agg\outs\agg2\r9\SJ\direct\20220922\diffs\SJ_r9_direct_0922_diffs.pkl',
                    'catMasks': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\cMasks\\SJ_r9_direct_0921_cMasks.pkl'
                    },
                'filter':{  
                    'agg': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\agg\\SJ_r9_filter_0921_agg.pkl',
                    'diffs': r'C:\LS\10_IO\2112_Agg\outs\agg2\r9\SJ\filter\20220922\diffs\SJ_r9_filter_0922_diffs.pkl',
                    'catMasks': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\cMasks\\SJ_r9_filter_0921_cMasks.pkl'
                }},
            'r10':
              {
                'direct':{  
                    'agg': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\agg\\SJ_r9_direct_0921_agg.pkl',
 
                    },
                'filter':{  
                    'agg': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\agg\\SJ_r9_filter_0921_agg.pkl',
 
                }
                
                }}
        

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
    from agg2.haz.scripts import UpsampleSessionXR as Session    
    #execute
    with Session(case_name=case_name,method=method,crs=crs, nodata=-9999, dsc_l=dsc_l, **kwargs) as ses:
        stat_d = dict()
        log = ses.logger
        #=======================================================================
        # build aggregated layers
        #=======================================================================
        """todo: add something similar for the layers"""
        if not 'agg' in fp_d:
            fp_d['agg'] = ses.run_agg(dem_fp, wse_fp, method=method)
            ses._clear()
            
        
        #pre-processed base rasters
        base_fp_d = pd.read_pickle(fp_d['agg']).iloc[0, :].to_dict()
        
 
        if not 'aggXR' in fp_d:
            fp_d['aggXR'] = ses.build_downscaled_aggXR(pd.read_pickle(fp_d['agg']))
            
            
            
        #=======================================================================
        # build difference grids
        #=======================================================================
        #=======================================================================
        # if not 'diffs' in fp_d:
        #     fp_d['diffs'] = ses.run_diffs(fp_d['agg'])
        #     ses._clear()
        #=======================================================================
        if not 'diffXR' in fp_d:
            fp_d['diffXR'] = ses.run_diffsXR(fp_d['aggXR'])
            
            
        #=======================================================================
        # category masks
        #=======================================================================
        if not ('catMasks' in fp_d) or not ('cmXR' in fp_d):            
            fp_d['cmXR'], fp_d['catMasks'] = ses.run_catMasks(base_fp_d['dem'], base_fp_d['wse'], write_tif=False)            
            ses._clear()
            
            

 
        
        if not 'XR' in fp_d:
            fp_d['XR'] = ses.run_merge_XR([fp_d[k] for k in ['diffXR', 'cmXR', 'aggXR']]) 
            
        return 
        #=======================================================================
        # prob of TP per cell
        #=======================================================================
        #ses.run_pTP(fp_d['aggXR'], fp_d['catMasks'], write=False)
 
 
        #=======================================================================
        # assemble vrts
        #=======================================================================
        for k, v in fp_d.items():
            if not v.endswith('.pkl'): continue 
            vrt_d = ses.run_vrts(v, resname=k, logger=log.getChild(k))  
 
            
        #=======================================================================
        # build agg stats
        #=======================================================================
        if 'agg' in fp_d:
            stat_d['s2'] = ses.run_stats(fp_d['agg'], fp_d['catMasks'])
            stat_d['s1'] = ses.run_stats_fine(fp_d['agg'], fp_d['catMasks'])
        
        #=======================================================================
        # build difference stats
        #=======================================================================
        if 'diffs' in fp_d:
            stat_d['diffs'] =ses.run_diff_stats(fp_d['diffs'], fp_d['catMasks'])
        
        ses.concat_stats(stat_d, write=False)

        #=======================================================================
        # wrap
        #=======================================================================
        
        log.info('finished w/ array picks \n%s'%pprint.pformat(fp_d, width=10, indent=3, compact=True, sort_dicts =False))
        log.info('finished w/ stat picks \n%s'%pprint.pformat(stat_d, width=30, indent=3, compact=True, sort_dicts =False))
        out_dir = ses.out_dir
        
        
    print('finished to %s'%out_dir)

    return fp_d, stat_d

def run_haz_stats(xr_fp,
                  proj_d=None,
                  case_name='SJ',
                 **kwargs):
    """hazard/raster stat compute from xarray"""
 
    
    #===========================================================================
    # extract parametesr
    #===========================================================================
    #project data   
    if proj_d is None: 
        proj_d = proj_lib[case_name] 
 
    crs = CRS.from_epsg(proj_d['EPSG'])
    
    
    
    #===========================================================================
    # run model
    #===========================================================================
    from agg2.haz.scripts import UpsampleSessionXR as Session
    
    out_dir = os.path.join(
            pathlib.Path(os.path.dirname(xr_fp)).parents[3],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
                    'hstats', today_str)
    #execute
    with Session(case_name=case_name,crs=crs, nodata=-9999, out_dir=out_dir, **kwargs) as ses: 
        log = ses.logger
        
        with xr.open_dataset(xr_fp, engine='netcdf4',chunks='auto', decode_coords="all") as ds:
            scale_l = ds[ses.idxn].values.tolist()
            log.info(f'loaded {ds.dims} from {os.path.basename(xr_fp)}' + 
                 f'\n    coors: {list(ds.coords)}' + 
                 f'\n    data_vars: {list(ds.data_vars)}' + 
                 f'\n    crs:{ds.rio.crs}' + 
                 f'\n    scales:{scale_l}'
                 )
            assert ds.rio.crs == ses.crs
            #=======================================================================
            # compute special stats
            #=======================================================================
            #ses.run_TP(ds)
            
            #=======================================================================
            # get basic stats
            #=======================================================================
            d = dict()
            for base in ['s2', 's1']:
                d[base] = ses.run_statsXR(ds, base=base, logger=log.getChild(base))
            
            rdx = pd.concat(d, axis=1, names=['base'])
            """
            view(rdx.T)
            """
            
            ofp = os.path.join(out_dir, f'{ses.fancy_name}_stats.pkl')
            rdx.to_pickle(ofp)
            
            log.info(f'wrote {str(rdx.shape)} to \n    {ofp}')
        
 
            
            
        
        
        
        
        
        
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




def SJ_run(run_name='r9',method='direct',**kwargs):
    return run_haz_agg2(case_name='SJ', fp_d = res_fp_lib[run_name][method], method=method, run_name=run_name, **kwargs)


def SJ_dev(run_name='t',method='direct',**kwargs):
    return run_haz_agg2(case_name='SJ', fp_d = {}, method=method, run_name=run_name, 
                        dsc_l=[1,  2**3, 2**4],
                        bbox = sgeo.box(2492040.000, 7436320.000, 2492950.000, 7437130.000),
                        **kwargs)

if __name__ == "__main__": 
    start = now()
    #===========================================================================
    # with Client(threads_per_worker=4, n_workers=1) as client:
    #     
    #     print(f' running dask client {client.dashboard_link}')
    #     #webbrowser.open(client.dashboard_link)
    #===========================================================================
        
    #SJ_dev()
    # SJ_run(method='direct',
    #             #dsc_l=[1,2**5,  2**7],
    #             run_name='r10'
    #            )
    #===========================================================================
    
    run_haz_stats(
        r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\direct\20220924\mXR\SJ_t_direct_0924_mXR.nc'
        )
 
 
    print('finished in %.2f'%((now()-start).total_seconds())) 