'''
Created on Sep. 10, 2022

@author: cefect
'''
import os, copy, datetime, logging
import numpy as np
import numpy.ma as ma
import pandas as pd
from pandas.testing import assert_index_equal

idx= pd.IndexSlice

#from definitions import max_cores
#from multiprocessing import Pool

#from agg2.haz.coms import coldx_d, cm_int_d
from hp.pd import append_levels
from hp.basic import lib_iter
 
#===============================================================================
# setup matplotlib----------
#===============================================================================
cm = 1/2.54
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
 
#font
matplotlib.rc('font', **{
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8})
 
 
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(10,10),
    'legend.title_fontsize':'large'
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)

#from agg2.haz.da import UpsampleDASession
from agg2.expo.scripts import ExpoSession
from agg2.haz.run_da import log_dxcol
from agg2.coms import Agg2DAComs
from hp.plot import view


def now():
    return datetime.datetime.now()


class ExpoDASession(ExpoSession, Agg2DAComs):
 
    def __init__(self,scen_name='expo_da',  **kwargs):
 
 
 
        super().__init__(scen_name=scen_name, logfile_duplicate=False,
                          **kwargs)
 
    
    def join_layer_samps(self,fp_lib,
                         dsc_df=None,
                         **kwargs):
        """assemble resample class of assets
        
        this is used to tag assets to dsc for reporting.
        can also compute stat counts from this
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('lsampsK',  subdir=True,ext='.pkl', **kwargs)
        
        res_d = dict()
        for method, d1 in fp_lib.items():
            d = dict()
            for layName, fp in d1.items():
                if not layName in ['wd', 'wse', 'dem']: continue        
 
                d[layName] = pd.read_pickle(fp)
                """
                pd.read_pickle(fp).hist()
                """
                
            #wrap method
            res_d[method] = pd.concat(d, axis=1).droplevel(0, axis=1) #already a dx
            
        #wrap
        dx1 =  pd.concat(res_d, axis=1, names=['method']).sort_index(axis=1)
 
        
        return dx1
    
    def build_samps(self, fp_lib, **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('bsamps',  subdir=True,ext='.pkl', **kwargs)
        # get the rsc for each asset and scale
        
        # join the simulation results (and clean up indicides
        samp_dx_raw = self.join_layer_samps(fp_lib)
        """
    
        view(samp_dx_raw.head(100))
        
        samp_dx.loc[:, idx['filter', 'wse', (1, 8)]].hist()
        
        """
        #=======================================================================
        # compute exposures
        #=======================================================================
        """1=wet"""
        wet_bxcol = samp_dx_raw.loc[:, idx[:, 'wse', :]].droplevel('layer', axis=1).notna().astype(int)
        #check false negatives
        wet_falseNegatives_bx = (wet_bxcol.subtract(
            wet_bxcol.loc[:, idx[:, 1]].droplevel('scale', axis=1)
            ) == -1)
        if wet_falseNegatives_bx.any().any():
            log.info('got %i False Negative exposures\n ' % (
                    wet_falseNegatives_bx.sum().sum(), 
                    #wet_falseNegatives[wet_falseNegatives != 0]
                    ))
        #merge
        samp_dx = pd.concat([
            samp_dx_raw, 
            pd.concat({'expo':wet_bxcol}, axis=1, names=['layer']).swaplevel('layer', 'method', axis=1)
            ], axis=1).sort_index(axis=1, sort_remaining=True)
            
        log.info(f'built {str(samp_dx.shape)}')
        return samp_dx
    
    def build_stats(self, samp_dx, dsc_df, **kwargs):
        """calc sample stats
        
        unlike Haz, we are computing the stats during data analysis
        
        s2 is computed against the matching samples
        s1 is computed against the baseline samples
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('bstats',  subdir=True,ext='.pkl', **kwargs)
        pdc_kwargs = dict(axis=1, names=['base'])
        
        lvls = ['base', 'method', 'layer', 'metric', 'dsc']
        
        def sort_dx(dx):
            return dx.reorder_levels(lvls, axis=1).sort_index(axis=1, sort_remaining=True)
        
        #=======================================================================
        # baseline values
        #=======================================================================
        samp_base_dx = samp_dx.loc[:, idx[:, :, 1]].droplevel('scale', axis=1)
        ufunc_d = {'expo':'sum', 'wd':'mean', 'wse':'mean'}
        
        #get baseline stats
        d = dict()
        for layName, stat in ufunc_d.items():
            dxi = samp_base_dx.loc[:, idx[:, layName]].droplevel('layer', axis=1)
            d[layName] = pd.concat({stat:getattr(dxi, stat)()}, axis=1, names='metric')
        
        #expand to match
        s1_sdxi = pd.concat(d, axis=1, names=['layer']).unstack().rename(1).to_frame().T.rename_axis('scale')        
        s1_sdx = sort_dx(pd.concat(
            {'full':pd.concat({'s1':s1_sdxi, 's2':s1_sdxi, 's12':pd.DataFrame(0.0, index=s1_sdxi.index, columns=s1_sdxi.columns)}, axis=1, names=['base'])
            }, axis=1, names=['dsc']))
        
        
        #=======================================================================
        # add GRANULAR
        #=======================================================================
        #samp_s12_dx = samp_dx.subtract(samp_base_dx, axis=1) 
        
        #=======================================================================
        #ZONAL stats
        #=======================================================================
        res_d = dict()
        for scale, col in dsc_df.items():
            """loop and compute stats with different masks"""
            d = dict()
            #===================================================================
            # #s1 (calc against base)
            #===================================================================
            mdex = pd.MultiIndex.from_frame(samp_base_dx.index.to_frame().reset_index(drop=True).join(col.rename('dsc')))
            dxi = pd.DataFrame(samp_base_dx.values, index=mdex, columns=samp_base_dx.columns)
            d['s1'] = self.get_dsc_stats2(dxi, ufunc_d=ufunc_d)
            
            #===================================================================
            # s2
            #===================================================================
            dxi = pd.DataFrame(samp_dx.loc[:, idx[:, :, scale]].values, index=mdex, columns=samp_base_dx.columns)
            d['s2'] = self.get_dsc_stats2(dxi, ufunc_d=ufunc_d)
            
            #===================================================================
            # s12
            #===================================================================
            dxi = pd.DataFrame(
                samp_dx.loc[:, idx[:, :, scale]].droplevel('scale', axis=1).sub(samp_base_dx).values, 
                index=mdex, columns=samp_base_dx.columns)
            d['s12'] = self.get_dsc_stats2(dxi, ufunc_d=ufunc_d)
            #===================================================================
            # wrap
            #===================================================================
            res_d[scale] = pd.concat(d, axis=1, names=['base'])
        
        #wrap
        sdx1 = sort_dx(pd.concat(res_d, axis=1, names=['scale']).stack('scale').unstack('dsc'))
        sdx2 = pd.concat([sdx1, s1_sdx]).sort_index() #add baseline (scale=1)
        
        """
        view(sdx2.drop('filter', level='method', axis=1).loc[:, idx[:, :, 'expo', :, 'full']])
        """
        #=======================================================================
        # compute residuals
        #=======================================================================
        srdx = sdx2['s2'].subtract(sdx2['s1']) #ggregated stats
        
        #normalize
        base_serx1 = s1_sdxi.iloc[0, :].reorder_levels(['method', 'layer', 'metric']).sort_index()
        
 
        #combine everything
        sdx3 = pd.concat([sdx2, 
                          pd.concat({
                                's12A':srdx, 's12AN':srdx.divide(base_serx1, axis=1), 's12N':sdx2['s12'].divide(base_serx1, axis=1)
                                }, **pdc_kwargs)
                          ], axis=1)
        

        
        #=======================================================================
        # write
        #=======================================================================
        log_dxcol(log, sdx3)
        if write: 
            sdx3.to_pickle(ofp)
            log.info(f'wrote {str(sdx3.shape)} to \n    {ofp}')
            
        return sdx3 
    
    def build_combined(self,
                       fp_lib,
                       **kwargs):
        """combine hazard and exposure datasets"""
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('bc',  subdir=True,ext='.pkl', **kwargs)
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
        for aname in ['base', 'method', 'dsc']:
            if not np.array_equal(
                hmdex.unique(aname),
                amdex.unique(aname)
                ):
                raise AssertionError(f'bad match on {aname}')
        
        assert set(hmdex.unique('layer')).difference(amdex.unique('layer'))==set(), 'layer name mismatch'
        
        #===================================================================
        # join
        #===================================================================
        haz_dx1 = haz_dx.reorder_levels(expo_dx.columns.names, axis=1).droplevel((1,2))
        dx1 = pd.concat({'exp':expo_dx, 'haz':haz_dx1},names=['phase'], axis=1)
        
        log.info(f'merged haz and expo data to get {str(dx1.shape)} w/ coldex:\n    {dx1.columns.names}')
        
        #=======================================================================
        # write
        #=======================================================================
        log_dxcol(log, dx1)
        if write: 
            dx1.to_pickle(ofp)
            log.info(f'wrote {str(dx1.shape)} to \n    {ofp}')
        
        return dx1
    
    def get_dsc_stats2(self, raw_dx,
                       ufunc_d = {'expo':'sum', 'wd':'mean', 'wse':'mean'},
                       **kwargs):
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('dscStats',  subdir=True,ext='.pkl', **kwargs)
        
        gcols = ['method', 'layer']
        res_d = dict()
        for i, (gkeys, gdx) in enumerate(raw_dx.groupby(level=gcols, axis=1)):
            gkeys_d = dict(zip(gcols, gkeys))
            stat = ufunc_d[gkeys_d['layer']]
            
            #get zonal stats            
            grouper = gdx.groupby(level='dsc')            
            sdx = getattr(grouper, stat)()
            
            #get total stat
            sdx.loc['full', :] = getattr(gdx, stat)()
            
            res_d[i] = pd.concat({stat:sdx}, axis=1, names=['metric'])
 
        return pd.concat(list(res_d.values()), axis=1).reorder_levels(list(raw_dx.columns.names) + ['metric'], axis=1)
    
 
    def plot_grid_d(self, 
                    data_lib,post_lib,
                    title=None, colorMap=None, color_d=None,
                    matrix_kwargs=dict(figsize=(17*cm, 18*cm) , set_ax_title=False, add_subfigLabel=True),
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
                                    sharey='row',sharex='all',  
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
    
    
        

    
    
    
    
