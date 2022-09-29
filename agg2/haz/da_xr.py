'''
Created on Sep. 28, 2022

@author: cefect

data analysis on xarrays
'''
import numpy as np
 
import pandas as pd
import os, copy, datetime, gc
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from agg2.haz.da import UpsampleDASession
from agg2.haz.scripts import UpsampleSessionXR
import dask.delayed
from dask.distributed import Client
 

class Session_haz_da_rast(UpsampleSessionXR, UpsampleDASession):
    def get_kde_df(self, xar_raw,dim='scale',

                          **kwargs):
        """plot a set of gaussian kdes
        
        Parameters
        -----------
        dim: str
            dimension of DataArray to build kde progression on
            
        Todo
        -----------
        use dask delayed and build in parallel?
        
        just partial zones?
        """
        #=======================================================================
        # defautlts
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('kde_df',  ext='.pkl', **kwargs)
        idxn = self.idxn
        scale_l = xar_raw[dim].values.tolist()
        
        log.info(f'building {len(scale_l)} on {xar_raw.shape}')
        
        xar1 = xar_raw.squeeze(drop=True).reset_coords( #drop band
            names=['spatial_ref'], drop=True
            ).transpose(idxn, ...)  #put scale first for iterating
 

        
        #=======================================================================
        # loop and build values
        #=======================================================================
 
        
        @dask.delayed
        def get_vals(xari):
            
            dar = xari.stack(rav=list(xari.coords)).dropna('rav').data
            kde = scipy.stats.gaussian_kde(dar, 
                                               bw_method='scott',
                                               weights=None, #equally weighted
                                               )
            
            xvals = np.linspace(dar.min()+.01, dar.max(), 200)
            
            return pd.concat({'x':pd.Series(xvals), 'y':pd.Series(kde(xvals))})
        
        
            
        def get_all_vals(xar):
            d = dict()
            for i, (scale, xari) in enumerate(xar.groupby(idxn, squeeze=False)):
                d[scale] = get_vals(xari)
 
            return d
        
        #=======================================================================
        # with Client(threads_per_worker=1, n_workers=6) as client:
        # 
        #     print(f' opening dask client {client.dashboard_link}')
        #=======================================================================
        log.info(f'executing get_all_vals on  {xar1.shape}')
                 
        def concat(d):
            df = pd.concat(d, axis=1, names=['scale'])
            df.index.set_names(['dim', 'coord'], inplace=True)
            return df.unstack('dim') 
        
        o=dask.delayed(concat)(get_all_vals(xar1))
        #df.visualize(filename=os.path.join(out_dir, 'dask_visualize.svg'))
        #d = dask.compute(get_all_vals(xar1))
        
        df = o.compute()
        
        #=======================================================================
        # wrap
        #=======================================================================
        df.to_pickle(ofp)
        log.info(f'wrote {str(df.shape)} to file\n    {ofp}')
        
        
        return df
        
    def plot_gaussian_set(self,dxcol, 
                                 fig=None, 
                          fig_kwargs = dict(figsize=(10,5)),
                          output_fig_kwargs=dict(),
                          color_d=None,
                          mean_line=True,
                          **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('kde',  ext='.svg', **kwargs)
        idxn =self.idxn
        scale_l = dxcol.columns.unique(idxn).tolist()
        #=======================================================================
        # setup plot
        #=======================================================================
        if fig is None:
            fig, ax = plt.subplots(**fig_kwargs)
        else:
            ax = fig.gca()
            
 
        if color_d is None:
            cmap = plt.cm.get_cmap(name='copper')            
            color_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(
                zip(scale_l, np.linspace(0, 1, len(scale_l)))).items()}
            
            
        #=======================================================================
        # loop and plot
        #=======================================================================
        
        for i, (scale, gdf) in enumerate(dxcol.groupby(idxn, axis=1)):
 
            log.info(f'    {i+1}/{len(scale_l)} scale={scale}')
            xar, yar = gdf.T.values
  
            
            ax.plot(xar, yar, 
                color=color_d[scale], label=scale, linewidth=0.5,
                    **kwargs)
            
            """need the full data for this...."""
 #==============================================================================
 #            #vertical mean line
 #            if not mean_line is None:
 # 
 #                """should be the same"""
 #                ax.axvline(xar.mean(), color=color_d[scale], linestyle='dashed')
 #==============================================================================
                
            log.debug(f'finished in {scale}')
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished on {len(scale_l)}')
        """
        plt.show()
        """
        #=======================================================================
        # output
        #=======================================================================
        if write:
            return self.output_fig(fig, ofp=ofp, logger=log, **output_fig_kwargs)
        else:
            return ax
        