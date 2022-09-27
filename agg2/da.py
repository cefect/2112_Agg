'''
Created on Sep. 27, 2022

@author: cefect

data analysis on hazard and exposure combined
'''
import numpy as np
 
import pandas as pd
from hp.basic import lib_iter
from agg2.haz.da import log_dxcol
from agg2.expo.da import ExpoDASession
import matplotlib.pyplot as plt
cm = 1/2.54

class CombinedDASession(ExpoDASession):
    def build_combined(self,
                       fp_lib,
                       names_l=None,
                       **kwargs):
        """combine hazard and exposure datasets"""
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('bc',  subdir=True,ext='.pkl', **kwargs)
        if names_l is None: names_l=self.names_l
        #=======================================================================
        # load data-------
        #=======================================================================
        haz_dx = pd.read_pickle(fp_lib['haz'])
        log.info(f'loaded hazard data {str(haz_dx.shape)} w/ coldex:\n    {haz_dx.columns.names}')
        
        expo_dx = pd.read_pickle(fp_lib['exp'])
        log.info(f'loaded expo data {str(expo_dx.shape)} w/ coldex:\n    {expo_dx.columns.names}')
        
        
        #===================================================================
        # check consistency
        #===================================================================
        assert np.array_equal(
            haz_dx.index.to_frame().reset_index(drop=True)['scale'].values,
            expo_dx.index.values
            )
        
        assert len(set(haz_dx.columns.names).symmetric_difference(expo_dx.columns.names))==0, 'column name mismatch'
        
        #chekc column values
        hmdex, amdex = haz_dx.columns, expo_dx.columns
        for aname in [
            #'base', 
            'method', 'dsc']:
            if not np.array_equal(
                hmdex.unique(aname),
                amdex.unique(aname)
                ):
                
                raise AssertionError(f'bad match on {aname}')
        
        assert set(hmdex.unique('layer')).difference(amdex.unique('layer'))==set(), 'layer name mismatch'
        
        #===================================================================
        # join
        #===================================================================
        haz_dx1 = haz_dx.reorder_levels(names_l, axis=1).droplevel((1,2))
        dx1 = pd.concat({'exp':expo_dx.reorder_levels(names_l, axis=1), 'haz':haz_dx1},names=['phase'], axis=1).sort_index(axis=1)
        
        log.info(f'merged haz and expo data to get {str(dx1.shape)} w/ coldex:\n    {dx1.columns.names}')
        
        #=======================================================================
        # write
        #=======================================================================
        log_dxcol(log, dx1)
        if write: 
            dx1.to_pickle(ofp)
            log.info(f'wrote {str(dx1.shape)} to \n    {ofp}')
        
        return dx1
    
    def plot_grid_d(self, 
                    data_lib,
                    title=None, colorMap=None, color_d=None,
                    matrix_kwargs=dict(figsize=(17*cm, 18*cm) , set_ax_title=False, add_subfigLabel=True,
                                       sharey='row',sharex='all',
                                       ),
                    plot_kwargs_lib={
                                          'full':{'marker':'x'},
                                          'DD':{'marker':'s', 'fillstyle':'none'},
                                          'WW':{'marker':'o', 'fillstyle':'full'},
                                          'WP':{'marker':'o', 'fillstyle':'top'},
                                          'DP':{'marker':'o', 'fillstyle':'bottom'},
                                          },
                    plot_kwargs={'linestyle':'solid', 'marker':'x', 'markersize':7, 'alpha':0.8},
                    output_fig_kwargs=dict(),
                    **kwargs):
        """grid plot from data in a dict. save post for caller"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('plot_gd',  subdir=False,ext='.svg', **kwargs)
        
        #=======================================================================
        # extract
        #=======================================================================
        # get first frame
        df0 = next(iter(next(iter(data_lib.values())).values()))
        
        keys_all_d = {'row':list(data_lib.keys()),
                      'col':list(next(iter(data_lib.values())).keys()),  # taking from first
                      'color':df0.columns.values.tolist(),
                      }
        
        # color
        color_key = df0.columns.name
        if color_d is None:
            color_d = self._get_color_d(color_key, keys_all_d['color'], colorMap=colorMap, color_d=color_d)
        
        # plot kwargs
        """here we populate with blank kwargs to ensure every series has some kwargs"""
        if plot_kwargs_lib is None: plot_kwargs_lib = dict()
        for k in keys_all_d['color']:
            if not k in plot_kwargs_lib:
                plot_kwargs_lib[k] = plot_kwargs
            else:
                plot_kwargs_lib[k] = {**plot_kwargs, **plot_kwargs_lib[k]}  # respects precedent
        
        log.info('plotting\n    rows:%s\n    cols:%s' % (keys_all_d['row'], keys_all_d['col']))
        #=======================================================================
        # setup figure
        #=======================================================================
        plt.close('all')
 
        fig, ax_d = self.get_matrix_fig(keys_all_d['row'], keys_all_d['col'],
                                    # figsize_scaler=4,                                    
                                    constrained_layout=False,
                                     
                                    fig_id=0,logger=log, **matrix_kwargs)
 
        if not title is None:
            fig.suptitle(title)
            
        #=======================================================================
        # loop and plot
        #=======================================================================

        #loop over the nested dictionary
        cnt=0
        for row_key, col_key, df in lib_iter(data_lib):
 
            #===================================================================
            # defaults
            #===================================================================
            log.debug(f'    on {row_key}.{col_key} for {str(df.shape)}')
            ax = ax_d[row_key][col_key]
            
            #check
            assert isinstance(df, pd.DataFrame)
            assert isinstance(df.columns, pd.Index)
            
            #===================================================================
            # plot each series (diff colors)
            #===================================================================
            for col_lab, col in df.items():
                ax.plot(col.index, col.values, color=color_d[col_lab],label=col_lab,**plot_kwargs_lib[col_lab])
            cnt+=1
        
        log.info('built %i plots'%cnt)
 
            
        
        return ax_d, keys_all_d