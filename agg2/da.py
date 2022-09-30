'''
Created on Sep. 27, 2022

@author: cefect

data analysis on hazard and exposure combined
'''
import os, pathlib, itertools, logging, string
import numpy as np
 
import pandas as pd
from hp.basic import lib_iter

from agg2.coms import log_dxcol, cat_mdex
from agg2.expo.da import ExpoDASession
 
idx = pd.IndexSlice
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
cm = 1/2.54

class PlotWrkr_4x4_matrix(object):
    
    def plot_4x4_matrix(self, dx1):
        """single figure w/ expo and hazard"""
        log = self.logger.getChild('p')
        mdex = dx1.columns
        #=======================================================================
        # separate data------
        #=======================================================================
    #['direct_haz', 'direct_exp', 'filter_haz', 'filter_exp']
    #col_keys = ['_'.join(e) for e in list(itertools.product(mdex.unique('method').values, ['haz', 'exp']))]
        col_keys = ['direct_haz', 'filter_haz', 'direct_exp', 'filter_exp']
        row_keys = ['wd_bias', 'wse_error', 'exp_area', 'vol']
    #empty container
        idx_d = dict()
        post_lib = dict()
        
        def set_row(method, phase):
            coln = f'{method}_{phase}'
            assert not coln in idx_d
            idx_d[coln] = dict()
            post_lib[coln] = dict()
            return coln
    #=======================================================================
    # #direct_haz
    #=======================================================================
        ea_base = 's12AN'
        method, phase = 'direct', 'haz'
        coln = set_row(method, phase)
        idx_d[coln]['wd_bias'] = idx[phase, 's12N', method, 'wd', 'mean', :]
        idx_d[coln]['wse_error'] = idx[phase, 's12', method, 'wse', 'mean', :]
        idx_d[coln]['exp_area'] = idx[phase, ea_base, method, 'wse', 'real_area', :]
        idx_d[coln]['vol'] = idx[phase, 's12AN', method, 'wd', 'vol', :]
    #=======================================================================
    # direct_exp
    #=======================================================================
        method, phase = 'direct', 'exp'
        coln = set_row(method, phase)
        idx_d[coln]['wd_bias'] = idx[phase, 's12N', method, 'wd', 'mean', :]
        idx_d[coln]['wse_error'] = idx[phase, 's12', method, 'wse', 'mean', :]
    #idx_d[coln]['exp_area']=    idx[phase, ea_base, method,'expo', 'sum', :]
    #idx_d[coln]['vol'] =        idx[phase, 's12N', method,'wd', 'vol', :]
    #=======================================================================
    # #filter_haz
    #=======================================================================
        method, phase = 'filter', 'haz'
        coln = set_row(method, phase)
        idx_d[coln]['wd_bias'] = idx[phase, 's12N', method, 'wd', 'mean', :]
        idx_d[coln]['wse_error'] = idx[phase, 's12', method, 'wse', 'mean', :]
        idx_d[coln]['exp_area'] = idx[phase, ea_base, method, 'wse', 'real_area', :]
        idx_d[coln]['vol'] = idx[phase, 's12AN', method, 'wd', 'vol', :]
    #=======================================================================
    # filter_exp
    #=======================================================================
        method, phase = 'filter', 'exp'
        coln = set_row(method, phase)
        idx_d[coln]['wd_bias'] = idx[phase, 's12N', method, 'wd', 'mean', :]
        idx_d[coln]['wse_error'] = idx[phase, 's12', method, 'wse', 'mean', :]
    #idx_d[coln]['exp_area'] =   idx[phase, ea_base, method,'expo', 'sum', :]
    #idx_d[coln]['vol'] =        idx[phase, 's12N', method,'wd', 'vol', :]
    #=======================================================================
    # check
    #=======================================================================
        cnt = 0
        for colk, d in idx_d.items():
            assert colk in col_keys, colk
            for rowk, idxi in d.items():
                assert rowk in row_keys, rowk
                assert len(dx1.loc[:, idxi]) > 0, f'bad on {rowk}.{colk}'
                cnt += 1
        
        log.info('built %i data selectors' % cnt)
    #=======================================================================
    # #collect
    #=======================================================================
        data_lib = {c:dict() for c in row_keys} #matching convention of get_matrix_fig() {row_key:{col_key:ax}}
        for colk in col_keys:
            for rowk in row_keys:
                if rowk in idx_d[colk]:
                    idxi = idx_d[colk][rowk]
                    data_lib[rowk][colk] = dx1.loc[:, idxi].droplevel(list(range(5)), axis=1)
        
    #===================================================================
    # plot------
    #===================================================================
        """such a custom plot no use in writing a function
    
    
    
    WSH:
    
        need to split axis
    
        direct
    
            why is normalized asset WD so high?
    
                because assets are biased to dry zones
    
                as the WD is smoothed, dry zones become wetter (and wet zones become drier)
    
    
    
    TODO:
    
    
    
    dx1['exp'].loc[:, idx[('s1', 's12', 's2', 's12N'), 'direct', 'wd', :, 'full']]
    
    """
        ax_d, keys_all_d = self.plot_grid_d(data_lib, 
            matrix_kwargs=dict(
                figsize=(17 * cm, 18 * cm), set_ax_title=False, add_subfigLabel=True, 
                sharey='none', sharex='all'))
        ylab_d = {}
            #===================================================================
            #   'wd_bias':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$',
            # 'wse_error':r'$\overline{WSE_{s2}}-\overline{WSE_{s1}}$',
            #   'exp_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
            #   'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$',
            #===================================================================
        ax_title_d = {
            'direct_haz':'full domain', 
            'direct_exp':'exposed domain', 
            'filter_haz':'full domain', 
            'filter_exp':'exposed domain'}
        ax_ylims_d = { #=================================================================
            'wd_bias':(-10, 10 ** 2)}
            # 'wse_error':(-1,15),
            # #'exp_area':(-1,20),
            # 'vol':(-0.2, .01),
            #=================================================================
        ax_yprec_d = {
            'wd_bias':0, 
            'wse_error':1, 
            'exp_area':0, 
            'vol':1}
        for row_key, col_key, ax in lib_iter(ax_d):
            ax.grid()
            #first col
            if col_key == col_keys[0]:
                if row_key in ylab_d:
                    ax.set_ylabel(ylab_d[row_key])
                else:
                    ax.set_ylabel(row_key)
                digits = ax_yprec_d[row_key]
                """not working... settig on all
    
            ax.yaxis.set_major_formatter(lambda x,p:f'%.{digits}f'%x)
    
            #ax.get_yaxis().set_major_formatter(lambda x,p:'{0:.{1}}'.format(x, digits))"""
                #last row
                #===============================================================
                # if row_key==row_keys[-1]:
                #     pass
                #===============================================================
            #first row
            if row_key == row_keys[0]:
                ax.set_title(ax_title_d[col_key])
                #ax.set_yscale('log')
                #last col
                if col_key == col_keys[-1]:
                    ax.legend()
            #last row
            if row_key == row_keys[-1]:
                ax.set_xlabel('resolution (m)')
                if 'exp' in col_key:
                    ax.axis('off')
                    for txt in ax.texts:
                        txt.set_visible(False)
        
            #all?
            #===================================================================
            # if row_key in ax_ylims_d:
            #     ax.set_ylim(ax_ylims_d[row_key])
            #===================================================================
    #add the titles
        fig = ax.figure
        fig.suptitle('direct', x=0.32)
        fig.text(0.8, 0.98, 'filter and subtract', size=matplotlib.rcParams['figure.titlesize'], ha='center')
    #fig.suptitle('filter and subtract', x=0.8)
    #=======================================================================
    # output
    #=======================================================================
        ofp = os.path.join(self.out_dir, f'{self.fancy_name}_matrix_combined.svg')
        self.output_fig(fig, ofp=ofp)
        
        return ofp



class PlotWrkr_4x4_subfigs(object):
    """complex 4x4 matrix plot with expo and haz"""
    def plot_4x4_subfigs(self, dx,
                         output_fig_kwargs=dict(),
                         **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('4x4_subfigs',  subdir=False,ext='.svg', **kwargs)
        #===========================================================================
        # build the figures
        #===========================================================================
        """
        TODO: dont use subfigures... create a custom gridspec
        """
        
        fig_master = plt.figure(num=0, constrained_layout=False, figsize=(17 * cm, 18 * cm))
        
        gs = fig_master.add_gridspec(1, 2, wspace=0, hspace=0, 
                                     width_ratios=(0.54, 0.46))
        
        
        #gs = gridspec.GridSpec(1, 2, figure=fig_master)
        
        fig_l = fig_master.add_subfigure(gs[0])        
        fig_r = fig_master.add_subfigure(
                    gs[1]#.subgridspec(4, 1, wspace=0, hspace=0)[0:3]
                    )
 
        #=======================================================================
        # fig_l, fig_r = fig_master.subfigures(nrows=1, ncols=2, 
        #                                      wspace=0.02, 
        #                                      squeeze=True)
        #=======================================================================
        
 
        
        
        mk_base=dict(set_ax_title=False, add_subfigLabel=False, 
                           constrained_layout=None, figsize=None,fig_id=None)
        
        #===========================================================================
        # add subfigs----
        #===========================================================================
        _, ax_dL = self.subfig_haz_4x2(dx, fig_l, mk_base)
        _, ax_dR = self.subfig_expo_3x2(dx, fig_r, mk_base)

        
        ax_d = {'left':ax_dL, 'right':ax_dR} #order matters
        """
        TODO:
            y labels
            yaxis precision
 
        """
        
        #=======================================================================
        # build positional container
        #=======================================================================
        ax_df = pd.DataFrame(index=pd.concat({k: pd.DataFrame.from_dict(d) for k,d in ax_d.items()}).stack().index,
                             columns=['row', 'col'])
        
        ax_df.index.set_names(['subfig', 'method', 'metric'], inplace=True)
        """easier to work with positional"""
        axi_d = dict()
        for j0, (subfig, d0) in enumerate(ax_d.items()):
 
            for i, (row_key, d1) in enumerate(d0.items()): #rows
                if not i in axi_d:
                    axi_d[i] = dict()                
                
                for j1, (col_key, ax) in enumerate(d1.items()):
                    j = (j0*2)+j1 #add the subaxis column to the subfig position
                    axi_d[i][j] = ax
                    
                    ax_df.loc[(subfig, col_key, row_key), 'col'] = j
                    ax_df.loc[(subfig, col_key, row_key), 'row'] = i
                    
                    #print(subfig, row_key, col_key)
        
        #clean up the map
        ax_pdx = ax_df.reset_index().pivot(index='row', columns='col')
        ax_pdx.columns.set_names(['meta', 'col'],   inplace=True)
        ax_pdx = ax_pdx.swaplevel(axis=1).stack('meta')
        

        #=======================================================================
        # post format
        #=======================================================================
        fig_l.suptitle('full domain', x=0.6)
        fig_r.suptitle('exposed domain', x=0.55)
        
        for i, d in axi_d.items():
            for j, ax in d.items():
                keys_d = ax_pdx.loc[idx[i, :], j].droplevel(0).to_dict() #get the labels here
                #print(keys_d)
                #===============================================================
                # subfig label                    
                #===============================================================
                ax.text(0.05, 0.95, 
                        '(%s%s)'%(list(string.ascii_lowercase)[j], i), 
                        transform=ax.transAxes, va='top', ha='left',
                        size=matplotlib.rcParams['axes.titlesize'],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                        )
                
                #===============================================================
                # axis labels
                #===============================================================
                if 'right' in keys_d['subfig']:
                    ax.get_yaxis().label.set_visible(False)
                    
        #=======================================================================
        # clean up first two rows
        #=======================================================================
        def ax_f(ax):
            ax.get_yaxis().set_ticklabels([])
            #ax.get_yaxis().set_visible(False)
        
        ax_f(axi_d[0][2])    
        ax_f(axi_d[1][2])
        
        #=======================================================================
        # x axis
        #=======================================================================
        axi_d[0][0].set_xlim((0,600))
        axi_d[0][2].set_xlim((0,600))
        #=======================================================================
        # change precision
        #=======================================================================
        def ax_f1(ax):
            ax.get_yaxis().set_major_formatter(lambda x,p:'%.1f'%x)
            
        ax_f1(axi_d[0][0])    
        ax_f1(axi_d[1][0])
        ax_f1(axi_d[2][0])                    
        #=======================================================================
        # remove dummy expo axis
        #=======================================================================

        def ax_hide(ax):
            ax.cla()
            #ax.axis('off')
            ax.set_xlabel('resolution (m)') #turn the label back on 
            ax.get_yaxis().set_visible(False)
            #ax.spines.clear()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
 
        ax_hide(axi_d[3][3])
        ax_hide(axi_d[3][2])
        
        #axi_d[1][3].legend()
        #=======================================================================
        # legend
        #=======================================================================
        #bots, tops, lefts, rights = axi_d[3][3].get_gridspec().get_grid_positions(fig_r)
        handles, labels = axi_d[0][3].get_legend_handles_labels() #get legned info 
        #fig_master.legend()
        fig_r.legend( handles, labels,   ncols=2,
                      bbox_to_anchor=(0.55, 0.25), loc='upper center',
                      title='resample case') #place it
        
 
        
        
        return self.output_fig(fig_master, ofp=ofp, logger=log, **output_fig_kwargs)
        """
        plt.close('all')
        plt.show()
        """
    def subfig_haz_4x2(self, dx, fig, mk_base, write=False):
        """hazard subfig
        
        
        adapted from haz.run_da
        """
        #===========================================================================
        # data prep
        #===========================================================================
        dxH = dx['haz']
        dxi = pd.concat([
                dxH.loc[:, idx['s12N', :, 'wd', :, 'mean']], 
                dxH.loc[:, idx['s12', :, 'wse', :, 'mean']], 
                dxH.loc[:, idx['s12AN', :, 'wse', :, 'real_area']], 
                dxH.loc[:, idx['s12AN', :, 'wd', :, 'vol']]
            ], axis=1).sort_index(axis=1)
            
        #cat layer and metric
        dxi.columns, mcoln = cat_mdex(dxi.columns, levels=['base', 'layer', 'metric']) 
        print(dxi.columns.unique(mcoln).tolist())
        row_l = ['s12N_wd_mean', 's12_wse_mean',  's12AN_wse_real_area', 's12AN_wd_vol']
        
        #stack into a series
        serx = dxi.stack(dxi.columns.names)
        assert set(serx.index.unique(mcoln)).symmetric_difference(dxi.columns.unique(mcoln)) == set()
     
        #===========================================================================
        # plot
        #===========================================================================
        
        return self.plot_matrix_metric_method_var(serx, 
                map_d={'row':mcoln, 'col':'method', 'color':'dsc', 'x':'scale'}, 
                row_l=row_l, 
                ylab_d={ 
                    's12N_wd_mean':r'$\frac{\overline{WSH_{s2} - WSH_{s1}}}{\overline{WSH_{s1}}}$',
                    's12_wse_mean':r'$\overline{WSE_{s2}-WSE_{s1}}$ (m)',
                    's12AN_wse_real_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$ or $\frac{\sum wet_{s2}-\sum wet_{s1}}{\sum wet_{s1}}$',
                    's12AN_wd_vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$',
                    
                    },
                ofp=os.path.join(self.out_dir, f'{self.fancy_name}_fig_haz_4x2_{mcoln}.svg'), 
                matrix_kwargs={**mk_base, **dict(fig=fig)}, 
                ax_lims_d={
                    'y':{
                        's12N_wd_mean':(-1.5, 3.0), 
                        's12AN_wse_real_area':(-0.2, 0.8),
                        's12N_wd_vol':(-0.3, 0.1), 
                        's12_wse_mean':(-1.0, 10.0)}}, 
                legend_kwargs=None,
                    write=write,)
    
    
    def subfig_expo_3x2(self, dx_raw, fig, mk_base, write=False):
        #===========================================================================
        # defaults
        #===========================================================================
        
        #===========================================================================
        # dataprep
        #===========================================================================
        dx = dx_raw['exp']
        self.log_dxcol(dx)
        """
        self.log_dxcol(dx)
        
        dx['s12A']
        
        view(dx.T)
        """
        #collect data slices
        dxi = pd.concat([
                dx.loc[:, idx['s12', :, 'wd', :, 'mean']], 
                dx.loc[:, idx['s12', :, 'wse', :, 'mean']], 
                dx.loc[:, idx['s12AN', :,'expo', :, 'sum']],
                dx.loc[:, idx['s12AN', :,'wd', :, 'sum']],   
     
            ], axis=1).sort_index(axis=1) 
        
        dxi.columns, mcoln = cat_mdex(dxi.columns, levels=['base', 'layer', 'metric']) #cat layer and metric
        print(dxi.columns.unique(mcoln).tolist())
     
        row_l = ['s12_wd_mean', 's12_wse_mean', 's12AN_expo_sum', 
                 's12AN_wd_sum'
                 ]
        
        #stack into a series
        serx = dxi.stack(dxi.columns.names)
        assert set(serx.index.unique(mcoln)).symmetric_difference(dxi.columns.unique(mcoln)) == set()
        #===========================================================================
        # plot 
        #===========================================================================
        
        return self.plot_matrix_metric_method_var(serx, 
                                      #title=baseName,
                                      row_l=row_l, 
                                      map_d = {'row':mcoln,'col':'method', 'color':'dsc', 'x':'scale'},
                                      ylab_d={
                                        #=======================================
                                        # 's12_wd_mean':'', 
                                        # 's12_wse_mean':'', 
                                        # 's12AN_expo_sum':r'$\frac{\sum wet_{s2}-\sum wet_{s1}}{\sum wet_{s1}}$',
                                        #=======================================
                                          },
                                     ofp=os.path.join(self.out_dir, f'{self.fancy_name}_fig_expo_3x2_{mcoln}.svg'),
                                      matrix_kwargs={**mk_base, **dict(fig=fig)}, 
                                      ax_lims_d = {'y':{
                                          's12_wd_mean':(-1.5,3.0), 
                                          's12_wse_mean':(-1, 10.0),
                                          's12AN_expo_sum':(-5, 20), #'expo':(-10, 4000)
                                          }},
                                      yfmt_func= lambda x,p:'%d'%x,
                                      legend_kwargs=None,
                                      write=write,
                                      )
        
class CombinedDASession(PlotWrkr_4x4_subfigs,PlotWrkr_4x4_matrix, ExpoDASession):
    
    def __init__(self,scen_name='daC',  **kwargs): 
        super().__init__(scen_name=scen_name,**kwargs)
        
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
    
if __name__ == "__main__":
    raise ImportError('???')