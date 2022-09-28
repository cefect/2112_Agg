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

class Session_haz_da_rast(UpsampleSessionXR, UpsampleDASession):
    def plot_gaussian_set(self, xar,dim='scale',fig=None, 
                          fig_kwargs = dict(figsize=(10,5)),
                          output_fig_kwargs=dict(),
                          color_d=None,
                          mean_line=True,
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
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('kde',  ext='.svg', **kwargs)
        
        scale_l = xar[dim].values.tolist()
        
        log.info(f'building {len(scale_l)} on {xar.shape}')
        
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
        # loop and build
        #=======================================================================
        for i, scale in enumerate(scale_l):
            log.info(f'    {i+1}/{len(scale_l)} scale={scale}')
            
            #get data
            xar1 = xar.loc[{dim:scale}].reset_coords(names=['spatial_ref'], drop=True) 
            
            #flatten and drop nulls
            xar2 = xar1.stack(rav=('x', 'y', 'band')).dropna('rav')
            """
            xari.values[0]
            """
            
            #build teh function            
            kde = scipy.stats.gaussian_kde(xar2, 
                                               bw_method='scott',
                                               weights=None, #equally weighted
                                               )
            
            #plot it
  
            xvals = np.linspace(xar2.min()+.01, xar2.max(), 200)
            ax.plot(xvals, kde(xvals), 
                color=color_d[scale], label=scale, linewidth=0.5,
                    **kwargs)
            
            #vertical mean line
            if not mean_line is None:
                ax.axvline(xar2.mean(), color=color_d[scale], linestyle='dashed')
                
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
            return fig
        