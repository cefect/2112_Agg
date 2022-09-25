'''
Created on Sep. 25, 2022

@author: cefect
'''
import os, pathlib, pprint, webbrowser
from hp.basic import get_dict_str, now, today_str
from rasterio.crs import CRS
import pandas as pd
import xarray as xr
from dask.distributed import Client
from definitions import proj_lib


xr_lib = {'r10':{
              'direct':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\direct\20220925\_xr',
              'filter':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\filter\20220925\_xr'
              }}

def run_haz_stats(xr_dir,
                  proj_d=None,
                  case_name='SJ',
                  fp_d=dict(),
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
            #pathlib.Path(os.path.dirname(xr_dir)).parents[0],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
            os.path.dirname(xr_dir),
                    'hstats', today_str)
    #execute
    with Session(case_name=case_name,crs=crs, nodata=-9999, out_dir=out_dir,xr_dir=xr_dir,method='hs', **kwargs) as ses: 
        log = ses.logger
        idxn = ses.idxn
        #config directory
        assert os.path.exists(xr_dir), xr_dir
        
 
        
        """we have 4 data_vars
        all coords and dims should be the same
        files are split along the 'scale' coord
        """
 
        #=======================================================================
        # load each subdir----
        #=======================================================================
        ds_d = dict()
        for dirpath, _, fns in os.walk(xr_dir):
            varName = os.path.basename(dirpath)            
            fp_l = [os.path.join(dirpath, e) for e in fns if e.endswith('.nc')]
            if len(fp_l)==0: continue
            ds_l = list()
            
            #load eaech
            for i, fp in enumerate(fp_l):
                ds_l.append(xr.open_dataset(fp, engine='netcdf4', chunks='auto',
                                            decode_coords="all"))
                
            ds_d[varName] = xr.concat(ds_l, dim=idxn)
                            
        #merge all the layers            
        ds = xr.merge(ds_d.values())
 
            
        log.info(f'loaded {ds.dims}'+
             f'\n    coors: {list(ds.coords)}'+
             f'\n    data_vars: {list(ds.data_vars)}'+
             f'\n    crs:{ds.rio.crs}'
             )
        assert ds.rio.crs == ses.crs, ds.rio.crs
        
 
        #=======================================================================
        # compute special stats-----
        #=======================================================================
        if not 's12_tp' in fp_d:
            fp_d['s12_TP'] = ses.run_TP_XR(ds)
        
        #difference
        if not 's12' in fp_d:
            s12_ds = ses.get_s12XR(ds)
            fp_d['s12'] =  ses.run_statsXR(s12_ds,base='s12',
                                        func_d={
                                            'wse':ses._get_diff_statsXR,#dome
                                            'wd':ses._get_diff_statsXR,
                                            })
         
        #=======================================================================
        # get basic stats
        #=======================================================================
        if not 's1' in fp_d or 's2' in fp_d:         
            for base in ['s2', 's1']:
                fp_d[base] = ses.run_statsXR(ds, base=base, logger=log.getChild(base))
            
        
         
        #=======================================================================
        # wrap
        #=======================================================================
        ds.close()
        
        d = {k:pd.read_pickle(fp) for k,fp in fp_d.items()}
        rdx = pd.concat(d, axis=1, names=['base'], sort=True)
        """
        view(rdx.T)
        """
         
        ofp = os.path.join(out_dir, f'{ses.fancy_name}_stats.pkl')
        rdx.to_pickle(ofp)
         
        log.info(f'wrote {str(rdx.shape)} to \n    {ofp}\n' +
                 pprint.pformat(fp_d, width=30, indent=3, compact=True, sort_dicts =False))
        
        return ofp
    
def SJ_run_h_stats(run_name='r10', method='direct'):
    return run_haz_stats(xr_lib[run_name][method], run_name=run_name, )
    
if __name__ == "__main__": 
    
    start = now()
    
          #=============================================================================
  #   with Client(
  #       #processes=True,
  #       #threads_per_worker=4, n_workers=3, memory_limit='2GB'
  #       ) as client:
  #        
  #       #get meta
  #       wrkr_cnt = len(client.scheduler_info()['workers'])
  #       for wName, wrkr in client.scheduler_info()['workers'].items():
  #           nthreads = wrkr['nthreads']
  #           break
  #        
  # 
  #       print(f' running dask client w/ {wrkr_cnt} workers and {nthreads} threads at {client.dashboard_link}')
  #       webbrowser.open(client.dashboard_link)
  #=============================================================================
    
    #run_haz_stats(r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\direct\20220925\_xr')
    SJ_run_h_stats(method='direct')
    
    print('finished in %.2f'%((now()-start).total_seconds())) 
        
   