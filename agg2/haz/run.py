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
        #=======================================================================
        # if not 'diffXR' in fp_d:
        #     fp_d['diffXR'] = ses.run_diffsXR(fp_d['aggXR'])
        #=======================================================================
            
            
        #=======================================================================
        # category masks
        #=======================================================================
        if not ('catMasks' in fp_d) or not ('cmXR' in fp_d):            
            fp_d['cmXR'], fp_d['catMasks'] = ses.run_catMasksXR(base_fp_d['dem'], base_fp_d['wse'], write_tif=False)            
            ses._clear()
            
            

 
        
        #=======================================================================
        # if not 'XR' in fp_d:
        #     fp_d['XR'] = ses.run_merge_XR([fp_d[k] for k in ['diffXR', 'cmXR', 'aggXR']]) 
        #=======================================================================
            
        return ses.xr_dir
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

def run_haz_agg2XR(method='direct',
            fp_d={},
            case_name = 'SJ',
            dsc_l=[1, 2**3, 2**5, 2**6, 2**7, 2**8, 2**9],
 
            proj_d=None,
                 **kwargs):
    """hazard/raster run for agg2 xarray"""
 
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
 
        if not 'agg' in fp_d:
            fp_d['agg'] = ses.run_agg(dem_fp, wse_fp, method=method)
            ses._clear()
            
        
        #pre-processed base rasters
        base_fp_d = pd.read_pickle(fp_d['agg']).iloc[0, :].to_dict()
        
 
        if not 'aggXR' in fp_d:
            fp_d['aggXR'] = ses.build_downscaled_aggXR(pd.read_pickle(fp_d['agg']))
            
        
        #=======================================================================
        # category masks
        #=======================================================================
        if not ('catMasks' in fp_d) or not ('cmXR' in fp_d):            
            fp_d['cmXR'], fp_d['catMasks'] = ses.run_catMasksXR(base_fp_d['dem'], base_fp_d['wse'], write_tif=False)            
            ses._clear()
            
 
        log.info(f'finished on \n    {ses.xr_dir}')
    return ses.xr_dir

def run_haz_stats(xr_dir,
                  proj_d=None,
                  case_name='SJ',
                 **kwargs):
    """hazard/raster stat compute from xarray"""
 
    
    #===========================================================================
    # extract parametesr
    #===========================================================================
    # project data   
    if proj_d is None: 
        proj_d = proj_lib[case_name] 
 
    crs = CRS.from_epsg(proj_d['EPSG'])
    
    #===========================================================================
    # run model
    #===========================================================================
    from agg2.haz.scripts import UpsampleSessionXR as Session
    
    out_dir = os.path.join(
            pathlib.Path(os.path.dirname(xr_dir)).parents[0],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
                    'hstats', today_str)
    #execute
    with Session(case_name=case_name,crs=crs, nodata=-9999, out_dir=out_dir, **kwargs) as ses: 
        log = ses.logger
        idxn = ses.idxn
        #config directory
        assert os.path.exists(xr_dir), xr_dir
        
 
        
        """we have 4 data_vars
        all coords and dims should be the same
        files are split along the 'scale' coord
        """
 
        #=======================================================================
        # load each subdir
        #=======================================================================
        ds_d = dict()
        for dirpath, _, fns in os.walk(xr_dir):
            varName = os.path.basename(dirpath)
            fp_l = [os.path.join(dirpath, e) for e in fns if e.endswith('.nc')]
            if not len(fp_l)>0:continue
            
            
            ds_d[varName] = xr.open_mfdataset(fp_l, 
                                #parallel=True,  #giving etCDF: Unknown file format
                                #engine='netcdf4',
                                data_vars='all', 
                                coords='all', 
                                combine="nested",
                                concat_dim=[idxn],
                                decode_coords="all",
                                combine_attrs='override',
                                chunks=1000,
                                )
            log.info(f'loaded {varName} from {len(fp_l)}')
            
        ds = xr.merge(ds_d.values())
            
        log.info(f'loaded {ds.dims}'+
             f'\n    coors: {list(ds.coords)}'+
             f'\n    data_vars: {list(ds.data_vars)}'+
             f'\n    crs:{ds.rio.crs}'
             )
        assert ds.rio.crs == ses.crs
        
        d = dict()
        #=======================================================================
        # compute special stats
        #=======================================================================
        d['s12_TP'] = ses.run_TP_XR(ds) 
         
        #=======================================================================
        # get basic stats
        #=======================================================================
         
        for base in ['s2', 's1']:
            d[base] = ses.run_statsXR(ds, base=base, logger=log.getChild(base))
         
        #=======================================================================
        # wrap
        #=======================================================================
        ds.close()
        
        rdx = pd.concat(d, axis=1, names=['base'], sort=True)
        """
        view(rdx.T)
        """
         
        ofp = os.path.join(out_dir, f'{ses.fancy_name}_stats.pkl')
        rdx.to_pickle(ofp)
         
        log.info(f'wrote {str(rdx.shape)} to \n    {ofp}')
        
        return ofp
        
        
 
            
            
        
        
        
        
        
        
def build_vrt(pick_fp = None,**kwargs):
    
    from agg2.haz.scripts import UpsampleSession as Session
    with Session(**kwargs) as ses:
        ses.run_vrts(pick_fp)
        
 




def SJ_run(run_name='r9',method='direct',**kwargs):
    return run_haz_agg2XR(case_name='SJ', fp_d = res_fp_lib[run_name][method], method=method, run_name=run_name, **kwargs)


def SJ_dev(run_name='t',method='direct',**kwargs):
    return run_haz_agg2XR(case_name='SJ', fp_d = {}, method=method, run_name=run_name, 
                        dsc_l=[1,  2**3, 2**4],
                        bbox = sgeo.box(2492040.000, 7436320.000, 2492950.000, 7437130.000),
                        **kwargs)

if __name__ == "__main__": 
    start = now()
    """seems to slow things down..."""
 #==============================================================================
 #    with Client(
 #        #processes=True,
 #        #threads_per_worker=4, n_workers=3, memory_limit='2GB'
 #        ) as client:
 #        
 #        #get meta
 #        wrkr_cnt = len(client.scheduler_info()['workers'])
 #        for wName, wrkr in client.scheduler_info()['workers'].items():
 #            nthreads = wrkr['nthreads']
 #            break
 #        
 # 
 #        print(f' running dask client w/ {wrkr_cnt} workers and {nthreads} threads at {client.dashboard_link}')
 #==============================================================================
        #webbrowser.open(client.dashboard_link)
        
    #xr_dir = SJ_dev(method='filter')
    #===========================================================================
    # nc_fp = SJ_run(method='direct',
    #             #dsc_l=[1,2**5,  2**7],
    #             run_name='r10'
    #            )
    #===========================================================================
    #run_haz_stats(r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\filter\20220925\_xr')
    #run_haz_stats(nc_fp)
 
 
    print('finished in %.2f'%((now()-start).total_seconds())) 
