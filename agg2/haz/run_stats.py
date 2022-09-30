'''
Created on Sep. 25, 2022

@author: cefect
'''
import os, pathlib, pprint, webbrowser, logging, sys
from hp.basic import get_dict_str, now, today_str
from rasterio.crs import CRS
import pandas as pd
import xarray as xr
#from dask.distributed import Client
import dask
import dask.config
from definitions import proj_lib
from hp.pd import view
idx = pd.IndexSlice

from agg2.haz.scripts import UpsampleSessionXR as Session

#===============================================================================
# setup logger
#===============================================================================
logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )

#===============================================================================
# result files
#===============================================================================

xr_lib = {'r10':{
              'direct':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\direct\20220925\_xr',
              'filter':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\filter\20220925\_xr'
              },
        'dev':{
            'direct':r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\direct\20220930\_xr',
            'filter':r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\filter\20220930\_xr',            
            }
        }

fp_lib = {'r10':{
    'filter':{  
       's12_TP': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\filter\20220925\hstats\20220926\tpXR\SJ_r10_hs_0926_tpXR.pkl',
       's12': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\filter\20220925\hstats\20220926\statsXR_s12\SJ_r10_hs_0926_statsXR_s12.pkl',
       #'s2': 'C:\\LS\\10_OUT\\2112_Agg\\outs\\agg2\\r10\\SJ\\filter\\20220925\\hstats\\20220925\\statsXR_s2\\SJ_r10_hs_0925_statsXR_s2.pkl',
       's1': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\filter\20220925\hstats\20220930\statsXR_s1\SJ_r10_hs_0930_statsXR_s1.pkl',
            },
    'direct':{
        #'s1':       r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\direct\20220925\hstats\20220925\statsXR_s1\SJ_r10_hs_0925_statsXR_s1.pkl',
        #'s2':       r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\direct\20220925\hstats\20220925\statsXR_s2\SJ_r10_hs_0925_statsXR_s2.pkl',
        's12_TP':   r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\direct\20220925\hstats\20220926\s12_TP\SJ_r10_hs_0926_s12_TP.pkl',
        's12':      r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\direct\20220925\hstats\20220926\statsXR_s12\SJ_r10_hs_0926_statsXR_s12.pkl',                
        }
    },
    'dev':{'filter':{}, 'direct':{}}
    }



#===============================================================================
# traditional stats-------
#===============================================================================
def run_haz_stats(xr_dir,
                  proj_d=None,
                  case_name='SJ',
                  fp_d=None,
                 **kwargs):
    """hazard/raster stat compute from xarray"""
 
    
    #===========================================================================
    # extract parametesr
    #===========================================================================
    if fp_d is None: fp_d=dict()
    # project data   
    if proj_d is None: 
        proj_d = proj_lib[case_name] 
 
    crs = CRS.from_epsg(proj_d['EPSG'])
    
    #===========================================================================
    # run model
    #===========================================================================
    
    
    out_dir = os.path.join(
            #pathlib.Path(os.path.dirname(xr_dir)).parents[0],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
            os.path.dirname(xr_dir),
                    'hstats', today_str)
    #execute
    with Session(case_name=case_name,crs=crs, nodata=-9999, out_dir=out_dir,xr_dir=xr_dir,method='hs', **kwargs) as ses: 
        log = ses.logger
        
        #build a datasource from the netcdf files
        ds = ses.get_ds_merge(xr_dir)
        
 
        #=======================================================================
        # compute special stats-----
        #=======================================================================
        if not 's12_TP' in fp_d:
            fp_d['s12_TP'] = ses.run_TP_XR(ds)
         
        #difference
        if not 's12' in fp_d:
            s12_ds = ses.get_s12XR(ds)
            fp_d['s12'] =  ses.run_statsXR(s12_ds,base='s12',
                                        func_d={
                                            'wse':ses._get_diff_wse_statsXR,
                                            'wd':ses._get_diff_wd_statsXR,
                                            })
         
        #=======================================================================
        # get basic stats
        #=======================================================================
        if not 's1' in fp_d:         
            fp_d['s1'] = ses.run_statsXR(ds, base='s1')
            
        if not 's2' in fp_d:         
            fp_d['s2'] = ses.run_statsXR(ds, base='s2')
        #=======================================================================
        # wrap
        #=======================================================================
        ds.close()
        
        d = {k:pd.read_pickle(fp) for k,fp in fp_d.items()}
        rdx = pd.concat(d, axis=1, names=['base'], sort=True)
        """
        rdx.loc[:, idx[:, :, :, ('posi_count', 'real_count')]]
        view(rdx.loc[:, idx[:, :, :, ('posi_count', 'real_count')]])
        view(rdx.T)
        """
         
        ofp = os.path.join(out_dir, f'{ses.fancy_name}_stats.pkl')
        rdx.to_pickle(ofp)
         
        log.info(f'wrote {str(rdx.shape)} to \n\n    {ofp}\n\n\n' +
                 pprint.pformat(fp_d, width=30, indent=3, compact=True, sort_dicts =False))
        
        return ofp
    
def SJ_run_h_stats(run_name='r10', method='direct'):
    return run_haz_stats(xr_lib[run_name][method], run_name=run_name,fp_d=fp_lib[run_name][method] )

#===============================================================================
# gaussian_kde--------
#===============================================================================
def SJ_compute_kde_run(
        run_name='r10',
        crs=None,
        case_name='SJ',
        **kwargs): 
 
    if crs is None:
        crs = CRS.from_epsg(proj_lib[case_name]['EPSG'])   
    
    return compute_kde(xr_lib[run_name], case_name=case_name, run_name=run_name,crs=crs,**kwargs)
        
        
def compute_kde(xr_dir_lib, 
                  **kwargs):
    """calculate kde plotting data"""

    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    """combines filter and direct"""
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(xr_dir_lib['direct'])).parents[1],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
        'da', 'haz', today_str)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir,logger=logging.getLogger('r'), **kwargs) as ses:
 
        idxn = ses.idxn
        log = ses.logger
        
        #=======================================================================
        # loop and build gaussian values on each
        #=======================================================================
 
        #build a datasource from the netcdf files
        d=dict()
        for method, xr_dir in xr_dir_lib.items():
            log.info(f'\n\non {method}\n\n')
            #===================================================================
            # load data
            #===================================================================        
            ds = ses.get_ds_merge(xr_dir)
            
            #===================================================================
            # plot gaussian of wd
            #===================================================================
            #data prep
            dar = ds['wd']
            dar1 = dar.where(dar[0]!=0)
            """
            dar1.plot(col='scale')
            """
            #get gausian data
            d[method] = ses.get_kde_df(dar1, logger=log.getChild(method), write=False)
            
        #=======================================================================
        # #merge
        #=======================================================================
        dxcol = pd.concat(d, axis=1, names=['method'])
        
        #cleanup
        xser = dxcol.loc[:, idx[:, :, 'x']].iloc[:,0].rename('x')
        
        dx1 = dxcol.loc[:, idx[:, :, 'y']].droplevel('dim', axis=1)
        
        dx1.index = pd.MultiIndex.from_frame(dx1.index.to_frame().join(xser))
        
 
        #=======================================================================
        # write
        #=======================================================================
        
        ofp = os.path.join(ses.out_dir, f'{ses.fancy_name}_kde_dxcol.pkl')
        dx1.to_pickle(ofp)
        log.info(f'wrote {dx1.shape} to\n    {ofp}')
        
    return ofp



            
 
    
if __name__ == "__main__": 
    
    start = now()
    
    #scheduler='single-threaded'
    scheduler='threads'
    with dask.config.set(scheduler=scheduler):
        print(scheduler)
        #print(pprint.pformat(dask.config.config, width=30, indent=3, compact=True, sort_dicts =False))
    
        #run_haz_stats(r'C:\LS\10_OUT\2112_Agg\outs\agg2\t\SJ\filter\20220925\_xr')
        SJ_run_h_stats(method='filter', run_name='r10')
        
        #SJ_compute_kde_run(run_name='r10')
 
    
    print('finished in %.2f'%((now()-start).total_seconds())) 
        
   