'''
Created on Jan. 18, 2022

@author: cefect



'''
#===============================================================================
# imports--------
#===============================================================================
import os, datetime, math, pickle, copy, random, pprint
import pandas as pd
import numpy as np

idx = pd.IndexSlice

import scipy.stats


from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo, \
    QgsWkbTypes

 
from hp.basic import set_info, get_dict_str

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from agg.coms.scripts import Session as agSession

import matplotlib


class Session(agSession):
    
    gcn = 'gid'
    scale_cn = 'scale'

    
    colorMap = 'cool'
    
    def __init__(self, 
                 name='hyd',
                 proj_lib={},
                 trim=True, #whether to apply aois
                 **kwargs):
        
        self.mindex_dtypes={
                 'studyArea':np.dtype('object'),
                 'id':np.dtype('int64'),
                 self.gcn:np.dtype('int64'), #both ids are valid
                 'grid_size':np.dtype('int64'), 
                 'event':np.dtype('O'),
                 self.scale_cn:np.dtype('int64'),           
                         }
        
        #===========================================================================
        # HANDLES-----------
        #===========================================================================
        #configure handles
        data_retrieve_hndls = {

            
            'finv_agg':{ #lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_finv_lib(**kwargs), #vlays need additional loading
                'build':lambda **kwargs: self.build_finv_agg(**kwargs),
                },
            
            'fgdir_dxind':{ #map of aggregated keys to true keys
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_finv_agg(**kwargs),
                },
            
            #gridded aggregation
            'finv_gPoly':{ #finv gridded polygons
                'compiled':lambda **kwargs:self.load_finv_lib(**kwargs),
                'build':lambda **kwargs: self.build_finv_gridPoly(**kwargs),
                },
            'finv_gPoly_id_dxind':{ #map of aggregated keys to true keys
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_finv_gridPoly(**kwargs),
                },
            
            'finv_sg_agg':{ #sampling geometry
                'compiled':lambda **kwargs:self.load_finv_lib(**kwargs), #vlays need additional loading
                'build':lambda **kwargs: self.build_sampGeo(**kwargs),
                },
            
            'tvals':{ #total asset values (series)
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_tvals(**kwargs),
                },
            
            'rsamps':{ #depth raster samples
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_rsamps(**kwargs),
                },
            
            'rloss':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_rloss(**kwargs),
                },
            'tloss':{ #total losses
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_tloss(**kwargs),
                },
            'errs':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_errs(**kwargs),
                }
            
            }
        
        super().__init__( 
                         data_retrieve_hndls=data_retrieve_hndls,name=name,
                         **kwargs)
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.proj_lib=proj_lib
        self.trim=trim
        
    
    #===========================================================================
    # PLOTTERS-------------
    #===========================================================================
    
    def plot_depths(self,
                    #data control
                    plot_fign = 'studyArea',
                    plot_rown = 'grid_size', 
                    plot_coln = 'event',                    
                    plot_zeros=False,
                    
                    #style control
                    xlims = (0,2),
                    ylims=(0,2.5),
                    calc_str = 'points',
                    
                    out_dir=None,
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_depths')
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
 
        serx = self.retrieve('rsamps')
        
        
        
        assert serx.notna().all().all(), 'drys should be zeros'

        """
        plt.show()
        self.retrieve('tvals')
        view(serx)
        """
        #=======================================================================
        # loop on studyAreas
        #=======================================================================
        
        log.info('on %i'%len(serx))
        
        
        
        res_d = dict()
        for i, (sName, gsx1r) in enumerate(serx.groupby(level=plot_fign)):
            plt.close('all')
            gsx1 = gsx1r.droplevel(plot_fign)
            mdex = gsx1.index
            
            fig, ax_d = self.get_matrix_fig(
                                    gsx1.index.unique(plot_rown).tolist(), #row keys
                                    gsx1.index.unique(plot_coln).tolist(), #col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='all', sharex='all', #everything should b euniform
                                    fig_id=i,
                                    set_ax_title=True,
                                    )
            
 
            
            
            for (row_key, col_key), gsx2r in gsx1r.groupby(level=[plot_rown, plot_coln]):
                #===============================================================
                # #prep data
                #===============================================================
                gsx2 = gsx2r.droplevel([plot_rown, plot_coln, plot_fign])
                
                if plot_zeros:
                    ar = gsx2.values
                else:
                    bx = gsx2>0.0
                    ar = gsx2[bx].values
                
                if not len(ar)>0:
                    log.warning('no values for %s.%s.%s'%(sName, row_key, col_key))
                    continue
                #===============================================================
                # #plot
                #===============================================================
                ax = ax_d[row_key][col_key]
                ax.hist(ar, color='blue', alpha=0.3, label=row_key, density=True, bins=30, range=xlims)
                
                #===============================================================
                # #label
                #===============================================================
                #get float labels
                meta_d = {'calc_method':calc_str, plot_rown:row_key,'wet':len(ar), 'dry':(gsx2<=0.0).sum(),
                           'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
                
 
                ax.text(0.5, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='blue')
                

                #===============================================================
                # styling
                #===============================================================                    
                #first columns
                if col_key == mdex.unique(plot_coln)[0]:
                    """not setting for some reason"""
                    ax.set_ylabel('density')
                
 
                #first row
                if row_key == mdex.unique(plot_rown)[0]:
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)
                    pass
                    #ax.set_title('event \'%s\''%(rlayName))
                    
                #last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel('depth (m)')
                    
                    
            fig.suptitle('depths for studyArea \'%s\' (%s)'%(sName, self.tag))
            #===================================================================
            # wrap figure
            #===================================================================
            res_d[sName]= self.output_fig(fig, out_dir=os.path.join(out_dir, sName), fname='depths_%s_%s'%(sName, self.longname))

        #=======================================================================
        # warp
        #=======================================================================
        log.info('finished writing %i figures'%len(res_d))
        
        return res_d

    def plot_tvals(self,
                    plot_fign = 'studyArea',
                    plot_rown = 'grid_size', 
                    #plot_coln = 'event',
 
                    
                    plot_zeros=True,
                    xlims = (0,200),
                    ylims=None,
                    
                    out_dir=None,
                    color='orange',
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_tvals')
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
 
        serx = self.retrieve('tvals')
        
        
        
        assert serx.notna().all().all(), 'drys should be zeros'

        """
        self.retrieve('tvals')
        view(serx)
        """
        #=======================================================================
        # loop on studyAreas
        #=======================================================================
        
        log.info('on %i'%len(serx))
        
        
        col_key = ''
        res_d = dict()
        for i, (sName, gsx1r) in enumerate(serx.groupby(level=plot_fign)):
            plt.close('all')
            gsx1 = gsx1r.droplevel(plot_fign)
            mdex = gsx1.index
            
            fig, ax_d = self.get_matrix_fig(
                                    gsx1.index.unique(plot_rown).tolist(), #row keys
                                    [col_key], #col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='all', sharex='all', #everything should b euniform
                                    fig_id=i,
                                    set_ax_title=True,
                                    )
            
 
            
            
            for row_key, gsx2r in gsx1r.groupby(level=plot_rown):
                #===============================================================
                # #prep data
                #===============================================================
                gsx2 = gsx2r.droplevel([plot_rown, plot_fign])
                bx = gsx2>0.0
                if plot_zeros:
                    ar = gsx2.values
                else:
                    
                    ar = gsx2[bx].values
                
                if not len(ar)>0:
                    log.warning('no values for %s.%s.%s'%(sName, row_key, ))
                    continue
                #===============================================================
                # #plot
                #===============================================================
                ax = ax_d[row_key][col_key]
                ax.hist(ar, color=color, alpha=0.3, label=row_key, density=True, bins=30, range=xlims)
                
                #label
                meta_d = {
                    plot_rown:row_key,
                    'cnt':len(ar), 'zeros_cnt':np.invert(bx).sum(),  'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
                
                txt = '\n'.join(['%s=%.2f'%(k,v) for k,v in meta_d.items()])
                ax.text(0.5, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='blue')
                

                #===============================================================
                # styling
                #===============================================================                    
                #first columns
                #===============================================================
                # if col_key == mdex.unique(plot_coln)[0]:
                #     """not setting for some reason"""
                #===============================================================
                ax.set_ylabel('density')
                
 
                #first row
                if row_key == mdex.unique(plot_rown)[0]:
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)
                    pass
                    #ax.set_title('event \'%s\''%(rlayName))
                    
                #last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel('total value (scale)')
                    
                    
            fig.suptitle('depths for studyArea \'%s\' (%s)'%(sName, self.tag))
            #===================================================================
            # wrap figure
            #===================================================================
            res_d[sName]= self.output_fig(fig, out_dir=os.path.join(out_dir, sName), fname='depths_%s_%s'%(sName, self.longname))

        #=======================================================================
        # warp
        #=======================================================================
        log.info('finished writing %i figures'%len(res_d))
        
        return res_d      
    """
    plt.show()
    """  
 
            
            
    def plot_tloss_bars(self, #barchart of total losses
                    dkey = 'tloss', #dxind to plot
                    lossType = 'tl', 
                    #plot_fign = 'studyArea',
                    plot_rown = 'studyArea', 
                    plot_coln = 'vid',
                    plot_colr = 'grid_size',
                    plot_bgrp = 'event',
                    
                    #plot style
                    ylabel=None,
                    colorMap=None,
                    yticklabelsF = lambda x,p: "{:,.0f}".format(x),
                    
                    ylims_d = {'Calgary':8e5, 'LMFRA':4e5, 'SaintJohn':1.5e5,'dP':0.5e5, 'obwb':6e5},
                    
                    add_label = True,
                   ):
        """
        matrix figure
            figure: studyAreas
                rows: grid_size
                columns: events
                values: total loss sum
                colors: grid_size
        
        """
        

        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_tloss')
        if colorMap is None: colorMap=self.colorMap
        if ylabel is None: ylabel = '%s sum'%lossType.upper()
        #=======================================================================
        # #retrieve child data
        #=======================================================================

        dx_raw = self.retrieve(dkey)
 
        log.info('on \'%s\' w/ %i'%(dkey, len(dx_raw)))
        
        #=======================================================================
        # setup data
        #=======================================================================
        """
        moving everything into an index for easier manipulation
        """
 
        
        #clip to total loss columns
        dxind1 = dx_raw.loc[:, idx[lossType, :]].droplevel(0, axis=1)
        dxind1.columns.name = 'vid'
        

        #promote column values to index
        dxser = dxind1.stack().rename(lossType)
 
        mdex = dxser.index
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
        """
        plt.show()
        """
 
        fig, ax_d = self.get_matrix_fig(
                                    mdex.unique(plot_rown).tolist(), #row keys
                                    mdex.unique(plot_coln).tolist(), #col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='row', sharex='row', #everything should b euniform
                                    fig_id=0,
                                    set_ax_title=False,
                                    )
        fig.suptitle('%s total on %i studyAreas (%s)'%(lossType.upper(), len(mdex.unique('studyArea')), self.tag))
        
        #get colors
        cvals = dxser.index.unique(plot_colr)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        #===================================================================
        # loop and plot
        #===================================================================
        for (row_key, col_key), gser1r in dxser.groupby(level=[plot_rown, plot_coln]):
            keys_d = dict(zip([plot_rown, plot_coln], (row_key, col_key)))
            
            #data setup
            gser1 = gser1r.droplevel([plot_rown, plot_coln])
 
            #subplot setup 
            ax = ax_d[row_key][col_key]
            
            
            #===================================================================
            # loop on colors
            #===================================================================
            for i, (ckey, gser2r) in enumerate(gser1.groupby(level=plot_colr)):
                keys_d[plot_colr] = ckey
                #get data
                gser2 = gser2r.droplevel(plot_colr)
 
                tlsum_ser = gser2.groupby(plot_bgrp).sum()
                
                #===============================================================
                # #formatters.
                #===============================================================
                #labels
 
                tick_label = ['e%i'%i for i in range(0,len(tlsum_ser))]
 
  
                #widths
                bar_cnt = len(mdex.unique(plot_colr))*len(tlsum_ser)
                width = 0.9/float(bar_cnt)
                
                #===============================================================
                # #add bars
                #===============================================================
                xlocs = np.linspace(0,1,num=len(tlsum_ser)) + width*i
                bars = ax.bar(
                    xlocs, #xlocation of bars
                    tlsum_ser.values, #heights
                    width=width,
                    align='center',
                    color=newColor_d[ckey],
                    label='%s=%s'%(plot_colr, ckey),
                    alpha=0.5,
                    tick_label=tick_label, 
                    )
                
                #===============================================================
                # add labels--------
                #===============================================================
                if add_label:
                    log.debug(keys_d)
                    assert plot_colr == 'grid_size'
                    if ckey==0:continue
                    
                    #===========================================================
                    # #calc errors
                    #===========================================================
                    d = {'pred':tlsum_ser}
                    #get trues
                    d['true'] = gser1r.loc[idx[0, keys_d['studyArea'],:, :, keys_d['vid']]].groupby('event').sum()
                    
                    d['delta'] = (tlsum_ser - d['true']).round(3)
                    
                    #collect
                    tl_df = pd.concat(d, axis=1)
                    
                    tl_df['relErr'] = (tl_df['delta']/tl_df['true'])
                
                    tl_df['xloc'] = xlocs
                    #===========================================================
                    # add as labels
                    #===========================================================
                    for event, row in tl_df.iterrows():
                        ax.text(row['xloc'], row['pred']*1.01, '%+.1f'%(row['relErr']*100),
                                ha='center', va='bottom', rotation='vertical',
                                fontsize=10,color='red')
                        
                    log.debug('added error labels \n%s'%tl_df)
                        
                    
                    
            #===============================================================
            # #wrap format subplot
            #===============================================================
            """
            fig.show()
            """
            del keys_d[plot_colr]
            ax.set_title(' & '.join(['%s:%s'%(k,v) for k,v in keys_d.items()]))
            #first row
            if row_key==mdex.unique(plot_rown)[0]:
 
                #last col
                if col_key == mdex.unique(plot_coln)[-1]:
                    ax.legend()
                    
            #first col
            if col_key == mdex.unique(plot_coln)[0]:
                ax.set_ylabel(ylabel)
                
                ax.get_yaxis().set_major_formatter(
                     matplotlib.ticker.FuncFormatter(yticklabelsF)
                     )
                
                if row_key in ylims_d:
                    ax.set_ylim((0, ylims_d[row_key]))
                    

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        self.output_fig(fig, fname='bars_%s_%s'%(lossType.upper(), self.longname))
        return
    
    def plot_terrs_box(self, #boxplot of total errors
                       
                    
                    #data control
                    dkey='errs',
                    ycoln = ('tl', 'delta'), #values to plot
                    plot_fign = 'studyArea',
                    plot_rown = 'event', 
                    plot_coln = 'vid',
                    plot_colr = 'grid_size',
                    #plot_bgrp = 'event',
                    
                    #plot style
                    ylabel=None,
                    colorMap=None,
                    add_text=True,
                    
                    out_dir=None,
                   ):
        """
        matrix figure
            figure: studyAreas
                rows: grid_size
                columns: events
                values: total loss sum
                colors: grid_size
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_terr_box')
        if colorMap is None: colorMap=self.colorMap
        if ylabel is None: ylabel = dkey
        if out_dir is None: out_dir=self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        log.info('on \'%s\' w/ %i'%(dkey, len(dx_raw)))
 
        
        #=======================================================================
        # setup data
        #=======================================================================
 

        #make slice
        dxser = dx_raw.loc[:, ycoln]
 
        #=======================================================================
        # loop on figures
        #=======================================================================
        for i, (fig_key, gser0r) in enumerate(dxser.groupby(level=plot_fign)):
            
            mdex = gser0r.index
            plt.close('all')
            
            fig, ax_d = self.get_matrix_fig(
                                        mdex.unique(plot_rown).tolist(), #row keys
                                        mdex.unique(plot_coln).tolist(), #col keys
                                        figsize_scaler=4,
                                        constrained_layout=True,
                                        sharey='all', 
                                        sharex='none', #events should all be uniform
                                        fig_id=i,
                                        set_ax_title=True,
                                        )
            
            s = '-'.join(ycoln)
            fig.suptitle('%s for %s:%s (%s)'%(s, plot_fign, fig_key, self.tag))
 
            """
            fig.show()
            """
        
            
            #===================================================================
            # loop and plot
            #===================================================================
            for (row_key, col_key), gser1r in gser0r.droplevel(plot_fign).groupby(level=[plot_rown, plot_coln]):
                
                #data setup
                gser1 = gser1r.droplevel([plot_rown, plot_coln])
     
                #subplot setup 
                ax = ax_d[row_key][col_key]
                
                #group values
                gd = {k:g.values for k,g in gser1.groupby(level=plot_colr)}
                
                #===============================================================
                # zero line
                #===============================================================
                ax.axhline(0, color='red')
 
                #===============================================================
                # #add bars
                #===============================================================
                boxres_d = ax.boxplot(gd.values(), labels=gd.keys(),meanline=True,
                           #boxprops={'color':newColor_d[rowVal]}, 
                           #whiskerprops={'color':newColor_d[rowVal]},
                           #flierprops={'markeredgecolor':newColor_d[rowVal], 'markersize':3,'alpha':0.5},
                            )
                
                #===============================================================
                # add extra text
                #===============================================================
                
                #counts on median bar
                for gval, line in dict(zip(gd.keys(), boxres_d['medians'])).items():
                    x_ar, y_ar = line.get_data()
                    ax.text(x_ar.mean(), y_ar.mean(), 'n%i'%len(gd[gval]), 
                            #transform=ax.transAxes, 
                            va='bottom',ha='center',fontsize=8)
                    
                    #===========================================================
                    # if add_text:
                    #     ax.text(x_ar.mean(), ylims[0]+1, 'mean=%.2f'%gd[gval].mean(), 
                    #         #transform=ax.transAxes, 
                    #         va='bottom',ha='center',fontsize=8, rotation=90)
                    #===========================================================
                                    

                    
                #===============================================================
                # #wrap format subplot
                #===============================================================
                ax.grid()
                #first row
                if row_key==mdex.unique(plot_rown)[0]:
                     
                    #last col
                    if col_key == mdex.unique(plot_coln)[-1]:
                        #ax.legend()
                        pass
                         
                #first col
                if col_key == mdex.unique(plot_coln)[0]:
                    ax.set_ylabel(ylabel)
                    
                #last row
                if row_key==mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel(plot_colr)
            #===================================================================
            # wrap fig
            #===================================================================
            log.debug('finsihed %s'%fig_key)
            self.output_fig(fig, fname='box_%s_%s'%(s, self.longname), 
                            out_dir=os.path.join(out_dir, fig_key))
                    

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
 
    def plot_errs_scatter(self, #scatter plot of error-like data
                    #data control   
                    dkey='errs',
                    
                    #lossType='rl',
                    ycoln = ('rl', 'delta'),
                    xcoln = ('depth', 'grid'),
                       
                    #figure config
                    folder_varn = 'studyArea',
                    plot_fign = 'event',
                    plot_rown = 'grid_size', 
                    plot_coln = 'vid',
                    plot_colr = None,
                    #plot_bgrp = 'event',
                    
                    
                    plot_vf=False, #plot the vf
                    plot_zeros=False,
                    
                    #axconfig
                    ylims = None,
                    xlims = None,
                    
                    #plot style
                    ylabel=None,
                    xlabel=None,
                    colorMap=None,
                    add_text=True,
                    
                    #outputs
                    fmt='png', transparent=False,
                    out_dir = None,
                   ):
        
        #raise Error('lets fit a regression to these results')
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_errs_scatter')
        if colorMap is None: colorMap=self.colorMap
        if ylabel is None: ylabel = '.'.join(ycoln)
        if xlabel is None: xlabel = '.'.join(xcoln)
        
        #=======================================================================
        # if plot_vf:
        #     assert lossType=='rl'
        #=======================================================================
            
        if plot_colr is None: plot_colr=plot_rown
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
 
        if plot_vf:
            vf_d = self.retrieve('vf_d')
        log.info('on \'%s\' for %s vs %s w/ %i'%(dkey, xcoln, ycoln, len(dx_raw)))
        
        #=======================================================================
        # prep data
        #=======================================================================
        #get slice specified by user
        dx1 = pd.concat([dx_raw.loc[:,ycoln], dx_raw.loc[:,xcoln]], axis=1)
        dx1.columns.set_names(dx_raw.columns.names, inplace=True)
        
 
        #=======================================================================
        # plotting setup
        #=======================================================================
        #get colors
        cvals = dx_raw.index.unique(plot_colr)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        
        """
        plt.show()
        """
        #=======================================================================
        # loop an study area/folders
        #=======================================================================
        
        for folder_key, gdx0 in dx1.groupby(level=folder_varn, axis=0):
 
            #=======================================================================
            # loop on figures
            #=======================================================================
            od = os.path.join(out_dir, folder_key, xlabel)
            plt.close('all')
        
            for i, (fig_key, gdx1) in enumerate(gdx0.groupby(level=plot_fign, axis=0)):
                keys_d = dict(zip([folder_varn, plot_fign], (folder_key, fig_key)))
                mdex = gdx1.index
                
                fig, ax_d = self.get_matrix_fig(
                                            mdex.unique(plot_rown).tolist(), #row keys
                                            mdex.unique(plot_coln).tolist(), #col keys
                                            figsize_scaler=4,
                                            constrained_layout=True,
                                            sharey='all', 
                                            sharex='all', #events should all be uniform
                                            fig_id=i,
                                            set_ax_title=False,
                                            )
                
                s = ' '.join(['%s:%s'%(k,v) for k,v in keys_d.items()])
                fig.suptitle('%s vs %s for %s'%(xcoln, ycoln, s))

            
                """
                fig.show()
                """
            
                #===================================================================
                # loop on axis row/column (and colors)----------
                #===================================================================
                for (row_key, col_key, ckey), gdx2 in gdx1.groupby(level=[plot_rown, plot_coln, plot_colr]):
                    keys_d.update(
                        dict(zip([plot_rown, plot_coln, plot_colr], (row_key, col_key, ckey)))
                        )
                    #skip trues
                    #===========================================================
                    # if ckey == 0:
                    #     continue 
                    #===========================================================
                    #subplot setup 
                    ax = ax_d[row_key][col_key]
 
                    #===============================================================
                    # prep data
                    #===============================================================
 
                    dry_bx = gdx2[xcoln]<=0.0
                    
                    if not plot_zeros:
                        xar, yar = gdx2.loc[~dry_bx, xcoln].values, gdx2.loc[~dry_bx, ycoln].values
                    else:
                        xar, yar = gdx2[xcoln].values, gdx2[ycoln].values

                    #===============================================================
                    # zero line
                    #===============================================================
                    ax.axhline(0, color='black', alpha=0.8, linewidth=0.5)
                    
                    #===============================================================
                    # plot function
                    #===============================================================
                    if plot_vf:
                        vf_d[col_key].plot(ax=ax,logger=log, set_title=False,
                                           lineKwargs=dict(
                            color='black', linestyle='dashed', linewidth=1.0, alpha=0.9)) 
     
    
                    
                    #===============================================================
                    # #add scatter plot
                    #===============================================================
                    ax.plot(xar, yar,
                               color=newColor_d[ckey], markersize=4, marker='x', alpha=0.8, 
                               linestyle='none',
                                   label='%s=%s'%(plot_colr, ckey))
 
                    #===========================================================
                    # add text
                    #===========================================================

                    if add_text:
                        meta_d = {'ycnt':len(yar), 
                                  'dry_cnt':dry_bx.sum(),
                                  'wet_cnt':np.invert(dry_bx).sum(),
                                  'y0_cnt':(yar==0.0).sum(),
                                  'ymean':yar.mean(), 'ymin':yar.min(), 'ymax':yar.max(),
                                  'xmax':xar.max(),
                              #'plot_zeros':plot_zeros,
                              }
                        
                        if ycoln[1] == 'delta':
                            meta_d['rmse'] = ((yar**2).mean())**0.5
                                            
                        txt = '\n'.join(['%s=%.2f'%(k,v) for k,v in meta_d.items()])
                        ax.text(0.1, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='black')
     
     
                    #===============================================================
                    # #wrap format subplot
                    #===============================================================
                    ax.set_title('%s=%s and %s=%s'%(
                         plot_rown, row_key,plot_coln, col_key))
                    
                    #first row
                    if row_key==mdex.unique(plot_rown)[0]:
                        
                        
                        #last col
                        if col_key == mdex.unique(plot_coln)[-1]:
                            pass
                             
                    #first col
                    if col_key == mdex.unique(plot_coln)[0]:
                        ax.set_ylabel(ylabel)
                        
                    #last row
                    if row_key==mdex.unique(plot_rown)[-1]:
                        ax.set_xlabel(xlabel)
                        
                    #loast col
                    if col_key ==mdex.unique(plot_coln)[-1]:
                        pass
                        #ax.legend()
                        
                #===================================================================
                # post format
                #===================================================================
                for row_key, ax0_d in ax_d.items():
                    for col_key, ax in ax0_d.items():
                        ax.grid()
                        
                        if not ylims is None:
                            ax.set_ylim(ylims)
                        
                        if not xlims is None:
                            ax.set_xlim(xlims)
                #===================================================================
                # wrap fig
                #===================================================================
                log.debug('finsihed %s'%fig_key)
                s = '_'.join(['%s'%(keys_d[k]) for k in [ folder_varn, plot_fign]])
                
                s2 = ''.join(ycoln) + '-' + ''.join(xcoln)
                
                self.output_fig(fig, out_dir=od, 
                                fname='scatter_%s_%s_%s'%( s2, s, self.longname.replace('_', '')),
                                fmt=fmt, transparent=transparent, logger=log)
            """
            fig.show()
            """
                    

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
    
    
    def plot_accuracy_mat(self, #matrix plot of accuracy
                    #data control   
                    dkey='errs',
                    lossType = 'tl',
 
                    
                    folder_varns = ['studyArea', 'event'], 
                    plot_fign = 'vid', #one raster:vid per plot
                    plot_rown = 'grid_size', 
                    plot_zeros=True,

                    #output control
                    out_dir = None,
                    fmt='png',
                    
 
 
                    
                    #plot style
                    binWidth=None,
                    colorMap=None,
                    lims_d = {'raw':{'x':None, 'y':None}} #control limits by column
                    #add_text=True,
                   ):
        
        """
        row1: trues
        rowx: grid sizes
        
        col1: hist of raw 'grid' values (for this lossType)
        col2: hist of delta values
        col3: scatter of 'grid' vs. 'true' values 
            """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_accuracy_mat.%s'%lossType)
        if colorMap is None: colorMap=self.colorMap
 
        if out_dir is None: out_dir = self.out_dir
        col_keys = ['raw', 'delta', 'correlation']
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        
        #slice by user
        dxind1 = dx_raw.loc[:, idx[lossType, ['grid', 'true', 'delta']]].droplevel(0, axis=1)
        """
        dx_raw.columns
        view(dx_raw)
        """
        #get colors
        cvals = dx_raw.index.unique(plot_rown)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        #=======================================================================
        # helpers
        #=======================================================================
        lim_max_d = {'raw':{'x':(0,0), 'y':(0,0)}, 'delta':{'x':(0,0), 'y':(0,0)}}
        def upd_lims(key, ax):
            #x axis
            lefti, righti = ax.get_xlim()
            leftj, rightj = lim_max_d[key]['x'] 
            
            lim_max_d[key]['x']  = (min(lefti, leftj), max(righti, rightj))
            
            #yaxis
            lefti, righti = ax.get_ylim()
            leftj, rightj = lim_max_d[key]['y'] 
            
            lim_max_d[key]['y']  = (min(lefti, leftj), max(righti, rightj))
        
        def set_lims(key, ax):
            if key in lims_d:
                if 'x' in lims_d[key]:
                    ax.set_xlim(lims_d[key]['x'])
                if 'y' in lims_d[key]:
                    ax.set_ylim(lims_d[key]['y'])
            
            upd_lims(key, ax)
        #=======================================================================
        # loop and plot----------
        #=======================================================================
        
        log.info('for \'%s\' w/ %i'%(lossType, len(dxind1)))
        for fkeys, gdxind1 in dxind1.groupby(level=folder_varns):
            keys_d = dict(zip(folder_varns, fkeys))
            
            
            
            for fig_key, gdxind2 in gdxind1.groupby(level=plot_fign):
                keys_d[plot_fign] = fig_key
                
                #setup folder
                od = os.path.join(out_dir, fkeys[0], fkeys[1], str(fig_key))
                """
                view(gdxind2)
                gdxind2.index
                fig.show()
                """
                log.info('on %s'%keys_d)
                #===============================================================
                # figure setup
                #===============================================================
                mdex = gdxind2.index
                plt.close('all')
                fig, ax_lib = self.get_matrix_fig(
                                            mdex.unique(plot_rown).tolist(), #row keys
                                            col_keys, #col keys
                                            figsize_scaler=4,
                                            constrained_layout=True,
                                            sharey='none', 
                                            sharex='none', #events should all be uniform
                                            fig_id=0,
                                            set_ax_title=True,
                                            )
                
                s = ' '.join(['%s-%s'%(k,v) for k,v in keys_d.items()])
                fig.suptitle('%s Accruacy for %s'%(lossType.upper(), s))
                
                #===============================================================
                # raws
                #===============================================================
                varn = 'grid'
                for ax_key, gser in gdxind2[varn].groupby(level=plot_rown):
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s'%(k,v) for k,v in keys_d.items()])
                    ax = ax_lib[ax_key]['raw']
                    self.ax_hist(ax,
                        gser, 
                        label=varn,
                        stat_keys = ['min', 'max', 'median', 'mean', 'std'],
                        style_d = dict(color=newColor_d[ax_key]),
                        binWidth=binWidth,
                        plot_zeros = plot_zeros,
                        logger=log.getChild(s1),
                        )
                    
                    #set limits
                    set_lims('raw', ax)
                    
                    
                #===============================================================
                # deltas
                #===============================================================
                varn='delta'
                for ax_key, gser in gdxind2[varn].groupby(level=plot_rown):
                    if ax_key == 0:continue
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s'%(k,v) for k,v in keys_d.items()])
                    
                    self.ax_hist(ax_lib[ax_key][varn],
                        gser, 
                        label=varn,
                        stat_keys = ['min', 'max', 'median', 'mean', 'std'],
                        style_d = dict(color=newColor_d[ax_key]),
                        binWidth=binWidth,
                        plot_zeros = plot_zeros,
                        logger=log.getChild(s1),
                        )
                    
                    upd_lims(varn, ax)
                #===============================================================
                # scatter
                #===============================================================
                for ax_key, gdxind3 in gdxind2.loc[:, ['grid', 'true']].groupby(level=plot_rown):
                    if ax_key == 0:continue
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s'%(k,v) for k,v in keys_d.items()])
                    
                    self.ax_corr_scat(ax_lib[ax_key]['correlation'],
                          
                          gdxind3['grid'].values, #x (first row is plotting gridded also)
                          gdxind3['true'].values, #y 
                          style_d = dict(color=newColor_d[ax_key]),
                          label='grid vs true',
                          
                          )
                    
                
                                      
                
                #=======================================================================
                # post formatting
                #=======================================================================
                """
                fig.show()
                """
                for row_key, d0 in ax_lib.items():
                    for col_key, ax in d0.items():
                        
                        #first row
                        if row_key==mdex.unique(plot_rown)[0]:
                            pass
                            
                            #last col
                            if col_key == col_keys[-1]:
                                pass
                            
                        #last row
                        if row_key==mdex.unique(plot_rown)[-1]:
                            #first col
                            if col_key == col_keys[0]:
                                ax.set_xlabel('%s (%s)'%(lossType, 'grid'))
                            elif col_key==col_keys[1]:
                                ax.set_xlabel('%s (%s)'%(lossType, 'delta'))
                            elif col_key==col_keys[-1]:
                                ax.set_xlabel('%s (%s)'%(lossType, 'grid'))
                                 
                        #first col
                        if col_key == col_keys[0]:
                            ax.set_ylabel('count')
                            ax.set_xlim(lim_max_d['raw']['x'])
                            ax.set_ylim(lim_max_d['raw']['y'])
                            
                        #second col
                        if col_key == col_keys[1]:
                            ax.set_ylim(lim_max_d['raw']['y'])
                            
                        #loast col
                        if col_key ==col_keys[-1]:
                            #set limits from first columns
                            col1_xlims = ax_lib[row_key]['raw'].get_xlim()
                            ax.set_xlim(col1_xlims)
                            ax.set_ylim(col1_xlims)
                            
                            if not row_key == 0:
                                
                                
                                
                                ax.set_ylabel('%s (%s)'%(lossType, 'true'))
                                #move to the right
                                ax.yaxis.set_label_position("right")
                                ax.yaxis.tick_right()
 
                #===============================================================
                # wrap fig
                #===============================================================
                s = '_'.join([str(e) for e in keys_d.values()])
                self.output_fig(fig, out_dir=od, 
                                fname='accuracy_%s_%s_%s'%(lossType, s, self.longname.replace('_', '')),
                                fmt=fmt, logger=log, transparent=False)
                
                
            #===================================================================
            # wrap folder
            #===================================================================
                
        #===================================================================
        # wrap
        #===================================================================
        log.info('finished')
        
        return
                
                    
    def ax_corr_scat(self, #correlation scatter plots on an axis
                ax,
                xar, yar, 
                label=None,
                
                #plot control
                plot_trend=True,
                plot_11=True,
                
                #lienstyles
                style_d = {},
                style2_d = { #default styles
                    'markersize':3.0, 'marker':'.', 'fillstyle':'full'
                    } ,
                
 
                logger=None,
 
                ):
        
        #=======================================================================
        # defaultst
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('ax_hist')
 
        #assert isinstance(stat_keys, list), label
        assert isinstance(style_d, dict), label
        #log.info('on %s'%data.shape)
        
 
        #=======================================================================
        # setup 
        #=======================================================================
        max_v = max(max(xar), max(yar))

        xlim = (min(xar), max(xar))
        #=======================================================================
        # add the scatter
        #=======================================================================
        ax.plot(xar, yar, linestyle='None',**style2_d, **style_d)
        """
        view(data)
        self.plt.show()
        """
        
        #=======================================================================
        # add the 1:1 line
        #=======================================================================
        if plot_11:
            #draw a 1:1 line
            ax.plot([0, max_v*10], [0, max_v*10], color='black',linewidth=0.5)
        
        #=======================================================================
        # add the trend line
        #=======================================================================
        if plot_trend:
            slope, intercept, rvalue, pvalue, stderr =  scipy.stats.linregress(xar, yar)
            
            pearson, pval = scipy.stats.pearsonr(xar, yar)
            
            
            x_vals = np.array(xlim)
            y_vals = intercept + slope * x_vals
            
            ax.plot(x_vals, y_vals, color='red', linewidth=0.5)
 
        #=======================================================================
        # get stats
        #=======================================================================
        
 
        
        stat_d = {
                'count':len(xar), 
                  #'LR.slope':round(slope, 3),
                  #'LR.intercept':round(intercept, 3),
                  #'LR.pvalue':round(slope,3),
                  #'pearson':round(pearson, 3), #just teh same as rvalue
                  'r value':round(rvalue,3),
                   #'max':round(max_v,3),
                   }

            
        #dump into a string
        annot = label +'\n'+ '\n'.join(['%s=%s'%(k,v) for k,v in stat_d.items()])
        
        anno_obj = ax.text(0.1, 0.9, annot, transform=ax.transAxes, va='center')
        
 
        #=======================================================================
        # add grid
        #=======================================================================
        
        ax.grid()
        
        return stat_d
                    
    def ax_hist(self, #set a histogram on the axis
                ax,
                data_raw, 
 
                label='',
                style_d = {}, 
                stat_keys = [],
 
                plot_zeros=False,
                binWidth=None,
 

                logger=None,
 
                ):
        #=======================================================================
        # defaultst
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('ax_hist')
        assert isinstance(data_raw, pd.Series), label
        assert isinstance(stat_keys, list), label
        assert isinstance(style_d, dict), label
        log.debug('on \'%s\' w/ %s'%(label, str(data_raw.shape)))
        
        #=======================================================================
        # setup  data
        #=======================================================================
        assert data_raw.notna().all().all()
        assert len(data_raw)>0
 
        dcount = len(data_raw) #length of raw data
        
        #handle zeros
        bx = data_raw <=0
        
        if bx.all():
            log.warning('got all zeros!')
            return {}
        
        if plot_zeros: 
            """usually dropping for delta plots"""
            data = data_raw.copy()
        else:
            data = data_raw.loc[~bx]
 
        
        if data.min()==data.max():
            log.warning('no variance')
            return {}
        #=======================================================================
        # #add the hist
        #=======================================================================
        assert len(data)>0
        if not binWidth is None:
            try:
                bins=np.arange(data.min(), data.max()+binWidth, binWidth)
            except Exception as e:
                raise Error('faliled to get bin dimensions w/ \n    %s'%e)
        else:
            bins=None
        
        histVals_ar, bins_ar, patches = ax.hist(
            data, 
            bins=bins, 
            stacked=False, label=label, 
            alpha=0.9, **style_d)
 
        #check
        assert len(bins_ar)>1, '%s only got 1 bin!'%label
 

        #=======================================================================
        # format ticks
        #=======================================================================
        #ax.set_xticklabels(['%.1f'%value for value in ax.get_xticks()])
            
        #===================================================================
        # #add the summary stats
        #===================================================================
        """
        plt.show()
        """

        bin_width = round(abs(bins_ar[1]-bins_ar[0]), 3)

 
        stat_d = {
            **{'count':dcount, #real values count
               'zeros (count)':bx.sum(), #pre-filter 
               'bin width':bin_width,
               #'bin_max':int(max(histVals_ar)),
               },
            **{k:round(getattr(data_raw, k)(), 3) for k in stat_keys}}
        
 
        #dump into a string
        annot = label +'\n'+ '\n'.join(['%s=%s'%(k,v) for k,v in stat_d.items()])
 
        anno_obj = ax.text(0.5, 0.8, annot, transform=ax.transAxes, va='center')
        
 
 
            
        return stat_d        
    
    #===========================================================================
    # ANALYSIS WRITERS---------
    #===========================================================================
    def write_loss_smry(self, #write statistcs on total loss grouped by grid_size, studyArea, and event
                    
                   #data control   
                    dkey='tloss',
                    #lossType='tl', #loss types to generate layers for
                    gkeys = [ 'studyArea', 'event','grid_size'],
                    
                    #output config
                    write=True,
                    out_dir=None,
                    ):
 
        """not an intermediate result.. jsut some summary stats
        any additional analysis should be done on the raw data
        """
        #=======================================================================
        # defaults
        #=======================================================================
        scale_cn = self.scale_cn
        log = self.logger.getChild('write_loss_smry')
        assert dkey=='tloss'
        if out_dir is None:
            out_dir = os.path.join(self.out_dir, 'errs')
            
 
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        #errors
        dx_raw = self.retrieve(dkey)
        
        """
        view(self.retrieve('errs'))
        view(dx_raw)
        """
 
        log.info('on %i for \'%s\''%(len(dx_raw), gkeys))
        #=======================================================================
        # calc group stats-----
        #=======================================================================
        rlib = dict()
        #=======================================================================
        # loss values
        #=======================================================================
        
        for lossType in dx_raw.columns.unique('lossType'):
            if lossType == 'expo':continue
            dxind1 = dx_raw.loc[:, idx[lossType, :]].droplevel(0, axis=1)
            #mdex = dxind1.index
            
            gbo = dxind1.groupby(level=gkeys)
            
            #loop and get each from the grouper
            d = dict()
            for statName in ['sum', 'mean', 'min', 'max']:
                d[statName] = getattr(gbo, statName)()
                
                
            #collect
            
            #=======================================================================
            # errors
            #=======================================================================
            """could also do this with the 'errs' data set... but simpler to just re-calc the totals here"""
            err_df = None
            for keys, gdf in gbo:
                keys_d = dict(zip(gkeys, keys))
                
                if keys_d['grid_size']==0: continue
                
                #get trues
                """a bit awkward as our key order has changed"""
                true_gdf = dxind1.loc[idx[0, keys_d['studyArea'], keys_d['event']], :]
     
                #calc delta (gridded - true)
                eser1 = gdf.sum() - true_gdf.sum()
                
     
                #handle results
                """couldnt figure out a nice way to handle this... just collecting in frame"""
                ival_ser = gdf.index.droplevel('gid').to_frame().reset_index(drop=True).iloc[0, :]
                
                eser2 = pd.concat([eser1, ival_ser])
                
                if err_df is None:
                    err_df = eser2.to_frame().T
                    
                else:
                    err_df = err_df.append(eser2, ignore_index=True)
            
            #collect
            d['delta'] = pd.DataFrame(err_df.loc[:, gdf.columns].values,
                index=pd.MultiIndex.from_frame(err_df.loc[:, gkeys]),
                columns = gdf.columns)
     
            
            rlib[lossType] = pd.concat(d, axis=1).swaplevel(axis=1).sort_index(axis=1)
        
        
 
        
        #=======================================================================
        # meta stats 
        #=======================================================================
        meta_d = dict()
        d=dict()
        dindex2 = dx_raw.loc[:, idx['expo', :]].droplevel(0, axis=1)
        
        d['count'] = dindex2['depth'].groupby(level=gkeys).count()
        
        #=======================================================================
        # depth stats
        #=======================================================================
        gbo = dindex2['depth'].groupby(level=gkeys)
        
        d['dry_cnt'] = gbo.agg(lambda x: x.eq(0).sum())
        
        d['wet_cnt'] = gbo.agg(lambda x: x.ne(0).sum())
 
 
        #loop and get each from the grouper
        for statName in ['mean', 'min', 'max', 'var']:
            d[statName] = getattr(gbo, statName)()
            
            
        meta_d['depth'] = pd.concat(d, axis=1)
        #=======================================================================
        # asset count stats
        #=======================================================================
        gbo = dindex2[scale_cn].groupby(level=gkeys)
        
        d=dict()
        
        d['mode'] = gbo.agg(lambda x:x.value_counts().index[0])
        for statName in ['mean', 'min', 'max', 'sum']:
            d[statName] = getattr(gbo, statName)()
            
 
        meta_d['assets'] = pd.concat(d, axis=1)
        
        #=======================================================================
        # collect all
        #=======================================================================
        rlib['meta'] = pd.concat(meta_d, axis=1)
        
        rdx = pd.concat(rlib, axis=1, names=['cat', 'var', 'stat'])
        
        #=======================================================================
        # write
        #=======================================================================
        log.info('finished w/ %s'%str(rdx.shape))
        if write:
            ofp = os.path.join(self.out_dir, 'lossSmry_%i_%s.csv'%(
                  len(dx_raw), self.longname))
            
            if os.path.exists(ofp): assert self.overwrite
            
            rdx.to_csv(ofp)
            
            log.info('wrote %s to %s'%(str(rdx.shape), ofp))
        
        return rdx
            
        """
        view(rdx)
        mindex.names
        view(dx_raw)
        """
    
    def write_errs(self, #write a points layer with the errors
                                       #data control   
                    dkey='errs',
                    
 
                    
                    #output config
                    #folder1_key = 'studyArea',
                    #folder2_key = 'event',
                    folder_keys = ['studyArea', 'event', 'vid'],
                    file_key = 'grid_size',
                    out_dir=None,
                    ):
        """
        folder: tloss_spatial
            sub-folder: studyArea
                sub-sub-folder: event
                    file: one per grid size 
        """
            
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('write_errs')
        assert dkey=='errs'
        if out_dir is None:out_dir = self.out_dir
            
        gcn = self.gcn
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        #errors
        dx_raw = self.retrieve(dkey)
        #mindex = dx_raw.index
        #names_d= {lvlName:i for i, lvlName in enumerate(dx_raw.index.names)}
        
        #get type
        #dx1 = dx_raw.loc[:, idx[lossType, :,:,:]].droplevel(0, axis=1)
        
        """
        dx_raw.index
        dx_raw.columns
        view(dxind_raw)
        view(dx_raw)
        view(tl_dxind)
        """
        #depths and id_cnt
        tl_dxind = self.retrieve('tloss')
        
        #geometry
        finv_agg_lib = self.retrieve('finv_agg')
        
        #=======================================================================
        # prep data
        #=======================================================================
        
 
        
        #=======================================================================
        # loop and write
        #=======================================================================
        meta_d = dict()
        #vf_d = self.retrieve('vf_d')
        log.info('on \'%s\' w/ %i'%(dkey, len(dx_raw)))
        
        #lvls = [names_d[k] for k in [folder1_key, folder2_key, file_key]]
        for i, (keys, gdx) in enumerate(dx_raw.groupby(level=folder_keys + [file_key])):
 
            keys_d = dict(zip(folder_keys + [file_key], keys))
            log.debug(keys_d)
            

                
            #===================================================================
            # retrieve spatial data
            #===================================================================
            #get vlay
            finv_vlay = finv_agg_lib[keys_d['studyArea']][keys_d['grid_size']]
 
            
            geo_d = vlay_get_geo(finv_vlay)
            fid_gid_d = vlay_get_fdata(finv_vlay, fieldn=gcn)
            #===================================================================
            # prepare data
            #===================================================================
            """layers only support 1d indexers... compressing 2d columsn here"""
            #get column values
            cdf = gdx.columns.to_frame().reset_index(drop=True)            
 
            #get flat frame
            gdf1 = pd.DataFrame(gdx.values,
                index=gdx.index.droplevel(list(keys_d.keys())),
                columns=cdf.iloc[:,0].str.cat(cdf.iloc[:,1], sep='.').values,
                ).reset_index()
            
 
            #reset to fid index
            gdf2 = gdf1.join(pd.Series({v:k for k,v in fid_gid_d.items()}, name='fid'), on='gid').set_index('fid')
            
            """
            view(gdf2)
            """
            #===================================================================
            # build layer
            #===================================================================
            layname = '_'.join([str(e).replace('_','') for e in keys_d.values()])
            vlay = self.vlay_new_df(gdf2, geo_d=geo_d, layname=layname, logger=log, 
                                    crs=finv_vlay.crs(), #session crs does not match studyAreas
                                    )
            
            #===================================================================
            # write layer
            #===================================================================
            #get output directory
            od = out_dir
            for fkey in [keys_d[k] for k in folder_keys]: 
                od = os.path.join(od, str(fkey))
 
            if not os.path.exists(od):
                os.makedirs(od)
                
            s = '_'.join([str(e) for e in keys_d.values()])
            ofp = os.path.join(od, self.longname + '_' + 'errs_' + s + '.gpkg') 
 
            ofp = self.vlay_write(vlay, ofp, logger=log)
            
            #===================================================================
            # wrap
            #===================================================================
            meta_d[i] = {**keys_d,
                **{ 'len':len(gdf2), 'width':len(gdf2.columns), 'ofp':ofp}
                }
            
        #=======================================================================
        # write meta
        #=======================================================================
        log.info('finished on %i'%len(meta_d))
        
        #write meta
        mdf = pd.DataFrame.from_dict(meta_d, orient='index')
        
        ofp = os.path.join(out_dir, '%s_writeErrs_smry.xls'%self.longname)
        with pd.ExcelWriter(ofp) as writer:       
            mdf.to_excel(writer, sheet_name='smry', index=True, header=True)
            
        return mdf
            
            
            
 
            
    def get_confusion_matrix(self, #wet/dry confusion
                             
                             #data control
                             dkey = 'errs',                             
                             group_keys = ['studyArea', 'event'],                             
                             pcn = 'grid_size', #label for prediction grouping
                             vid=None, #default is to take first vid
                             
                             #output control
                             out_dir=None,
                             ):
        
 
        """
        predicteds will always be dry (we've dropped these)
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_confusion_matrix')
        if out_dir is None: out_dir = os.path.join(self.out_dir, 'confusion_mat')
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        labs_d = {1:'wet', 0:'dry'}
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        
        #=======================================================================
        # prep data
        #=======================================================================
        if vid is None: vid = dx_raw.index.unique('vid')[0] #just taking one function
        dxind1 = dx_raw.loc[idx[:,:,:,vid,:], idx['depth', ['grid', 'true']]
                         ].droplevel(0, axis=1 #clean columns
                                     ).droplevel('vid', axis=0) #remove other vids
        
        
 
 
        # convert to binary
        bx = dxind1>0 #id wets
        
        dxind2 = dxind1.where(~bx, other=labs_d[1]).where(bx, other=labs_d[0]) #replace where false
        
        #=======================================================================
        # loop on each raster
        #=======================================================================
        mat_lib = dict()
        for keys, gdx0 in dxind2.groupby(level=group_keys):
            keys_d = dict(zip(group_keys, keys))
            
            if not keys[0] in mat_lib:
                mat_lib[keys[0]] = dict()
            
            cm_d = dict()
            #===================================================================
            # collect confusion on each grid_size
            #===================================================================
            for pkey, gdx1 in gdx0.groupby(level=pcn, axis=0):
                keys_d[pcn] = pkey
            
                gdf = gdx1.droplevel(level=list(keys_d.keys()), axis=0)
                #if keys_d['grid_size']==0: continue
                cm = confusion_matrix(gdf['true'].values, gdf['grid'].values, labels=list(labs_d.values()))
                
                cm_d[pkey] = pd.DataFrame(cm,index = labs_d.values(), columns=labs_d.values())
                
            #===================================================================
            # combine
            #===================================================================
            dxcol1 = pd.concat(cm_d, axis=1)
            
            #add predicted/true labels
            dx1 = pd.concat([dxcol1], names=['type', 'grid_size', 'exposure'], axis=1, keys=['predicted'])
            dx2 = pd.concat([dx1], keys=['true'])
            
            #store
            mat_lib[keys[0]][keys[1]] = dx2
 
            
        #===========================================================================
        # write
        #===========================================================================
        for studyArea, df_d in mat_lib.items():
            ofp = os.path.join(out_dir, '%s_confMat_%s.xls'%(studyArea, self.longname))
            
            with pd.ExcelWriter(ofp) as writer:
                for tabnm, df in df_d.items():
                    df.to_excel(writer, sheet_name=tabnm, index=True, header=True)
                    
            log.info('wrote %i sheets to %s'%(len(df_d), ofp))
            
        return mat_lib
            
    
 
    
    #===========================================================================
    # DATA CONSTRUCTION-------------
    #===========================================================================

    def build_errs(self, #get the errors (gridded - true)
                    dkey=None,
                     prec=None,
                     group_keys = ['grid_size', 'studyArea', 'event'],
                     write_meta=True,
 
                    ):
        """
        delta: grid - true
        errRel: delta/true
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_errs')
        assert dkey=='errs'
        if prec is None: prec=self.prec
        gcn = self.gcn
        scale_cn = self.scale_cn
 
        #=======================================================================
        # retriever
        #=======================================================================
        tl_dx = self.retrieve('tloss')
        
        tlnames_d= {lvlName:i for i, lvlName in enumerate(tl_dx.index.names)}
 
        fgdir_dxind = self.retrieve('fgdir_dxind')
        
        fgdir_dxind[0] = fgdir_dxind.index.get_level_values('id') #set for consistency
 
        #=======================================================================
        # group on index
        #=======================================================================
        log.info('on %s'%str(tl_dx.shape))
        res_dx = None
        for ikeys, gdx0 in tl_dx.groupby(level=group_keys, axis=0):
            ikeys_d = dict(zip(group_keys, ikeys))
            res_lib = {k:dict() for k in tl_dx.columns.unique(0)}
            #===================================================================
            # group on columns
            #===================================================================
            for ckeys, gdx1 in gdx0.groupby(level=gdx0.columns.names, axis=1):
                ckeys_d = dict(zip(gdx0.columns.names, ckeys))
                
                log.debug('on %s and %s'%(ikeys_d, ckeys_d))
 
 
                #===================================================================
                # get trues--------
                #===================================================================
                
                true_dx0 = tl_dx.loc[idx[0, ikeys_d['studyArea'], ikeys_d['event'], :],gdx1.columns]
                
                #===============================================================
                # join true keys to gid
                #===============================================================
                #relabel to true ids
                true_dx0.index.set_names('id', level=tlnames_d['gid'], inplace=True)
                
                #get true ids (many ids to 1 gid))
                id_gid_df = fgdir_dxind.loc[idx[ikeys_d['studyArea'], :], ikeys_d['grid_size']].rename(gcn).to_frame()
                id_gid_dx = pd.concat([id_gid_df], keys=['expo', 'gid'], axis=1)
                
                
                if not id_gid_dx.index.is_unique:
                    #id_gid_ser.to_frame().loc[id_gid_ser.index.duplicated(keep=False), :]
                    raise Error('bad index on %s'%ikeys_d)
                
                #join gids
                true_dxind1 = true_dx0.join(id_gid_dx, on=['studyArea', 'id']).sort_index().droplevel(0, axis=1)

                #===============================================================
                # summarize by type
                #===============================================================
                #get totals per gid
                gb = true_dxind1.groupby(gcn)
                if ckeys_d['lossType'] == 'tl': #true values are sum of each child
                    true_df0 = gb.sum()
                elif ckeys_d['lossType']=='rl': #true values are the average of family
                    true_df0 = gb.mean()
                elif ckeys_d['vid'] == 'depth':
                    true_df0 = gb.mean()
                elif ckeys_d['vid'] == scale_cn:    
                    true_df0 = gb.sum()
                else:
                    raise Error('bad lossType')

                assert true_df0.index.is_unique
                
                #expand index
                true_dx = pd.concat([true_df0], keys=['true', true_df0.columns[0]], axis=1) 
 
                #join back to gridded
                gdx2 = gdx1.join(true_dx, on=gcn, how='outer')
                
                #===========================================================
                # get gridded-------
                #===========================================================
 
                
                #check index
                miss_l = set(gdx2.index.get_level_values(gcn)).difference(true_dx.index.get_level_values(gcn))
                assert len(miss_l)==0, 'failed to join back some trues'

                """from here... we're only dealing w/ 2 columns... building a simpler calc frame"""
                
                gdf = gdx2.droplevel(1, axis=1).droplevel(group_keys, axis=0)
                gdf0 = gdf.rename(columns={gdf.columns[0]:'grid'})
                
                #===================================================================
                # error calcs
                #===================================================================
                #delta (grid - true)
                gdf1 = gdf0.join(gdf0['grid'].subtract(gdf0['true']).rename('delta'))
                
                #relative (grid-true / true)
                gdf2 = gdf1.join(gdf1['delta'].divide(gdf1['true']).fillna(0).rename('errRel'))
                
                 
 
                #===============================================================
                #clean
                #===============================================================
                #join back index
                rdxind1 = gdx1.droplevel(0, axis=1).join(gdf2, on=gcn).drop(gdx1.columns.get_level_values(1)[0], axis=1)
                
                #check
                assert rdxind1.notna().all().all()
                
                assert not ckeys[1] in res_lib[ckeys[0]]
                
                #promote
                res_lib[ckeys[0]][ckeys[1]] =rdxind1
                
  
            #===================================================================
            # wrap index loop-----
            #===================================================================
            
            #===================================================================
            # assemble column loops
            #===================================================================
            d2 = dict()
            names = [tl_dx.columns.names[1], 'metric']
            for k0, d in res_lib.items():
                d2[k0] = pd.concat(d, axis=1, names=names)
                
            rdx = pd.concat(d2, axis=1, names=[tl_dx.columns.names[0]] + names)
            
            #===================================================================
            # append
            #===================================================================
            if res_dx is None:
                res_dx = rdx
            else:
                res_dx = res_dx.append(rdx)
 
        #=======================================================================
        # wrap------
        #=======================================================================
        #=======================================================================
        # promote vid to index
        #=======================================================================
        """
        treating meta columns (id_cnt, depth) as 'vid'
        makes for more flexible data maniuplation
            although.. .we are duplicating lots of values now
        """
        
        res_dx1 = res_dx.drop('expo', level=0, axis=1)
        
        #promote column values to index
        res_dx2 = res_dx1.stack(level=1).swaplevel().sort_index()
        
        #pull out and expand the exposure
        exp_dx1 = res_dx.loc[:, idx['expo', :, :]].droplevel(0, axis=1)
        
        #exp_dx2 = pd.concat([exp_dx1, exp_dx1], keys = res_dx1.columns.unique(0), axis=1)
        
        #join back
        res_dx3 = res_dx2.join(exp_dx1, on=res_dx1.index.names).sort_index(axis=0)
        
 
        
        """
        view(res_dx3.droplevel('gid', axis=0).index.to_frame().drop_duplicates())
        view(res_dx3)
        """

        #===================================================================
        # meta
        #===================================================================
        gb = res_dx.groupby(level=group_keys)
         
        mdx = pd.concat({'max':gb.max(),'count':gb.count(),'sum':gb.sum()}, axis=1)
        
        if write_meta:
            ofp = os.path.join(self.out_dir, 'build_errs_smry_%s.csv'%self.longname)
            if os.path.exists(ofp):assert self.overwrite
            mdx.to_csv(ofp)
            log.info('wrote %s to %s'%(str(mdx.shape), ofp))
                
 


        log.info('finished w/ %s and totalErrors: \n%s'%(
            str(res_dx.shape), mdx))
 
        #=======================================================================
        # write
        #=======================================================================
 
        
        self.ofp_d[dkey] = self.write_pick(res_dx3,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                                   logger=log)

        return res_dx3
 
 
    def build_tloss(self, #get the total loss
                    dkey=None,
                    prec=2,
                    ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        scale_cn = self.scale_cn
        log = self.logger.getChild('build_tloss')
        assert dkey=='tloss'
        
        #=======================================================================
        # retriever
        #=======================================================================

        
        tv_serx = self.retrieve('tvals') #studyArea, id : grid_size : corresponding gid
        
        rl_dxind = self.retrieve('rloss')
        
        rlnames_d= {lvlName:i for i, lvlName in enumerate(rl_dxind.index.names)}
        
        #=======================================================================
        # join tval and rloss
        #=======================================================================
 
        dxind1 = rl_dxind.join(tv_serx, on=tv_serx.index.names)
 
        
        assert dxind1[scale_cn].notna().all()

        
        #re-org columns
        
        dx1 = pd.concat({
            'expo':dxind1.loc[:, ['depth', scale_cn]],
            'rl':dxind1.loc[:, rl_dxind.drop('depth', axis=1).columns]},
            axis=1, verify_integrity=True)
        
 
        
        #=======================================================================
        # calc total loss
        #=======================================================================
        #relative loss x scale
        tl_dx = dx1.loc[:, idx['rl', :]].multiply(
            dx1.loc[:, idx['expo', scale_cn]], axis=0
            ).round(prec)
            
        #rename the only level0 value        
        tl_dx.columns = tl_dx.columns.remove_unused_levels().set_levels(levels=['tl'], level=0)
        
        #join these in 
        dx2 = dx1.join(tl_dx)
 
        #set the names
        """these dont actually apply to the meta group.. but still nicer to have the names"""
        dx2.columns.set_names(['lossType', 'vid'], inplace=True)
        
        #=======================================================================
        # check
        #=======================================================================
        self.check_mindex(dx2.index)
 
        #=======================================================================
        # wrap
        #=======================================================================
        #reporting
        rdxind = dx2.loc[:, idx['tl', :]].droplevel(0, axis=1)
        mdf = pd.concat({
            'max':rdxind.groupby(level='grid_size').max(),
            'count':rdxind.groupby(level='grid_size').count(),
            'sum':rdxind.groupby(level='grid_size').sum(),
            }, axis=1)
        
        
        log.info('finished w/ %s and totalLoss: \n%s'%(
            str(dx2.shape), 
            #dxind3.loc[:,tval_colns].sum().astype(np.float32).round(1).to_dict(),
            mdf
            ))
        
        
 
 

        self.ofp_d[dkey] = self.write_pick(dx2,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                                   logger=log)

        return dx2
        
        
        
    
    def build_tvals(self, #get the total values on each asset
                    dkey=None,
                    prec=2,
                    tval_type='uniform', #type for total values
                    
                    **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_tvals')
        assert dkey=='tvals'
        if prec is None: prec=self.prec
        gcn = self.gcn
        scale_cn = self.scale_cn
        
        #=======================================================================
        # retriever
        #=======================================================================
        
        #fgdir_dxind = self.retrieve('fgdir_dxind') #studyArea, id : grid_size : corresponding gid
        finv_agg_lib = self.retrieve('finv_agg')
        #dxser = self.retrieve('rsamps')
 
        

        
        """
        view(dxser)
        view(finv_agg)
        view(fgdir_dxind)
        """
        #=======================================================================
        # build combined index from layers
        #=======================================================================
        """cleaner to build this in the shape/form expected by later functions
        contains ALL assets (even those without children or exposure)"""
        res_lib = dict()
        for studyArea, d1 in finv_agg_lib.items():
            res_d = dict()
            for grid_size, vlay in d1.items():
                df_raw = vlay_get_fdf(vlay)
                res_d[grid_size] = df_raw.set_index(gcn).iloc[:,0]
                
            res_lib[studyArea] = pd.concat(res_d)
            
        mindex = pd.concat(res_lib, names =['studyArea','grid_size', gcn], verify_integrity=True).sort_index().index
        
        """no... this is missing nulls. best to construct from layers
        #prepaer index (to match rsamps)
        fgdir_dxind[0] = fgdir_dxind.index.get_level_values('id') #add the un-gridded for consistency
        fg_dxind = fgdir_dxind.droplevel(1).stack().rename(gcn)
        mdf = fg_dxind.to_frame().set_index(gcn, append=True, drop=False).sort_index().index.to_frame().reset_index(drop=True).drop_duplicates(gcn)
        mdf = mdf.loc[:, ['grid_size', 'studyArea', gcn]] #reorder to match rsamps
        
        rser_dx = pd.Series(index=pd.MultiIndex.from_frame(mdf), name=scale_cn)"""
 
        #=======================================================================
        # calculate the asset scalers
        #=======================================================================
        rserx = getattr(self, 'tvals_' + tval_type)(mindex, **kwargs)
 

 
        #=======================================================================
        # check
        #=======================================================================
        assert rserx.name == scale_cn
        self.check_mindex(rserx.index)
 
 

        self.ofp_d[dkey] = self.write_pick(rserx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                                   logger=log)

        return rserx
    
    def tvals_rand(self,
                    mindex,
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('tvals_pnorm')
 
        scale_cn = self.scale_cn
        
        
        
        #=======================================================================
        # loop and calc true scales randomly
        #=======================================================================
        rserx = pd.Series(np.nan, index=mindex, name=scale_cn)
        levelNames = mindex.names[0:2]
        for keys, gser in pd.Series(index=mindex).groupby(level=levelNames):
            keys_d = dict(zip(levelNames, keys))
            
            if not keys_d['grid_size'] == 0: continue
            
            rser = pd.Series(np.random.random(len(gser)), index=gser.index, name=scale_cn)
            
            #set these
            rserx.loc[rser.index] = rser
 
                
        #=======================================================================
        # get aggregated scales
        #=======================================================================
        return self.agg_from_true(rserx, gfunc='sum', logger=log, **kwargs)
    

        
        
    def tvals_uniform(self, #get uniform tvals
                  mindex, #index for grouping results
 
                  **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('tvals_uni')
        #fgdir_dxind = self.retrieve('fgdir_dxind') #studyArea, id : grid_size : corresponding gid
        #gcn = self.gcn
        scale_cn = self.scale_cn
        
        #=======================================================================
        # loop and calc true scales randomly
        #=======================================================================
        rserx = pd.Series(np.nan, index=mindex, name=scale_cn)
        levelNames = mindex.names[0:2]
        for keys, gser in pd.Series(index=mindex).groupby(level=levelNames):
            keys_d = dict(zip(levelNames, keys))
            
            if not keys_d['grid_size'] == 0: continue
            
            rser = pd.Series(1, index=gser.index, name=scale_cn)
            
            #set these
            rserx.loc[rser.index] = rser
 
                
        #=======================================================================
        # get aggregated scales
        #=======================================================================
        return self.agg_from_true(rserx, gfunc='sum', logger=log, **kwargs)
            
        
    def agg_from_true(self, #aggregating grid values from true (grid_size=0) values
                      serx,
                      gfunc='sum', #groupby method to apply
                      logger=None,
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = self.logger.getChild('agg_from_true')
        fgdir_dxind = self.retrieve('fgdir_dxind') #studyArea, id : grid_size : corresponding gid
        gcn = self.gcn
        scale_cn = self.scale_cn
        #=======================================================================
        # loop and calc uinform scalers
        #=======================================================================
        rserx = None
        levelNames = serx.index.names[0:2]
        for keys, gser in serx.groupby(level=levelNames):
            keys_d = dict(zip(levelNames, keys))
            
            #===================================================================
            # un-gridded
            #===================================================================
            if keys_d['grid_size'] == 0:
                rser = gser.copy()
            
            #===================================================================
            # gridded
            #===================================================================
            else:
                #get lookups
                id_lookuop_serx = fgdir_dxind.loc[idx[keys_d['studyArea'], :], keys_d['grid_size']]
                id_lookuop_serx = id_lookuop_serx.rename(gcn).sort_index()
                
                #get trues
                t_serx = serx.loc[idx[keys[0], [0], :]].droplevel(1)
                t_serx.index.set_names('id', level=1, inplace=True) #rename the true gid to id
                
                #join true values to lookuop
                dxind1 = id_lookuop_serx.to_frame().join(t_serx)
                
                #collapse/group by gid
                gb = dxind1.groupby('gid')
                agg_df = getattr(gb, gfunc)()
                
 
                #get matching index
                for name in levelNames:
                    agg_df[name] = keys_d[name]
                    
                rser_raw = agg_df.set_index(levelNames, append=True).reorder_levels(gser.index.names).iloc[:,0]
            
                #update with these
                rser = gser.fillna(0).astype(int)
                rser.loc[rser_raw.index] = rser_raw
                
                
            #===================================================================
            # check
            #===================================================================
            assert rser.notna().all()
            rser = rser.sort_index()
            if not np.array_equal(gser.index.values, rser.index.values):
                raise Error('failed to get matching indicies on %s'%keys_d)
            assert isinstance(rser, pd.Series), keys_d
                
            #===================================================================
            # collect
            #===================================================================
            if rserx is None:
                rserx = rser
            else:
                rserx = rserx.append(rser)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.debug('finished on %i'%len(rserx))
        
        return rserx   

    
    def build_rloss(self, #calculate relative loss from rsamps on each vfunc
                    dkey=None,
                    prec=None, #precision for RL
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_rloss')
        assert dkey=='rloss'
        if prec is None: prec=self.prec
        
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        #depths
        #fgm_ofp_d, fgdir_dxind = self.get_finvg()
        dxser = self.retrieve('rsamps')
        
 
        log.debug('loaded %i rsamps'%len(dxser))
        
        #vfuncs
        vf_d = self.retrieve('vf_d')
        
        #=======================================================================
        # loop and calc
        #=======================================================================
        log.info('getting impacts from %i vfuncs and %i depths'%(
            len(vf_d), len(dxser)))
            
        res_d = dict()
        for i, (vid, vfunc) in enumerate(vf_d.items()):
            log.info('%i/%i on %s'%(i+1, len(vf_d), vfunc.name))
            
            ar = vfunc.get_rloss(dxser.values)
            
            assert ar.max() <=100, '%s returned some RLs above 100'%vfunc.name
            
            res_d[vid] = ar 
        
        
        #=======================================================================
        # combine
        #=======================================================================
        
        rdf = pd.DataFrame.from_dict(res_d).round(prec)
        
        rdf.index = dxser.index
        
        res_dxind = dxser.to_frame().join(rdf)
        
        log.info('finished on %s'%str(rdf.shape))
        
        self.check_mindex(res_dxind.index)
        
 
        #=======================================================================
        # write
        #=======================================================================
        self.ofp_d[dkey] = self.write_pick(res_dxind,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                                   logger=log)
        
        return res_dxind
    
    def build_finv_agg(self, #build aggregated finvs
                       dkey=None,
                       
                       #control aggregated finv type 
                       aggType='gridded',

                       **kwargs):
        """
        wrapper for calling more specific aggregated finvs (and their indexer)
            filepaths_dict and indexer are copied
            layer container is not
            
        only the compiled pickles for this top level are required
            (lower levels are still written though)
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_finv_agg')
        assert dkey in ['finv_agg', 'fgdir_dxind']
        
        log.info('building \'%s\' '%(aggType))
        
        #=======================================================================
        # retrive aggregated finvs------
        #=======================================================================
        """these should always be polygons"""
        
        #=======================================================================
        # select approriate dkeys
        #=======================================================================
        if aggType == 'gridded':
            finv_poly_dkey = 'finv_gPoly'
            finv_id_dxind = 'finv_gPoly_id_dxind'
            
 
        else:
            raise Error('not implemented')
        
        
        #=======================================================================
        # retrieve data        
        #=======================================================================
        finv_agg_lib = self.retrieve(finv_poly_dkey)
        fgdir_dxind = self.retrieve(finv_id_dxind)
        
        
        #=======================================================================
        # #get ofp_d
        #=======================================================================
        """special retrival to carry forward the data storage"""
        if finv_poly_dkey in self.ofp_d:
            pick_fp = self.ofp_d[finv_poly_dkey]
        else:
            pick_fp = self.compiled_fp_d[finv_poly_dkey]
            
        ofp_d = self.load_pick(fp=pick_fp)

        
        #=======================================================================
        # store results-------
        #=======================================================================
        #=======================================================================
        # finvs
        #=======================================================================
        dkey1 = 'finv_agg'
        self.ofp_d[dkey1] = self.write_pick(ofp_d, 
                           os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey1, self.longname)),logger=log)
        
        #save to data
        self.data_d[dkey1] = finv_agg_lib
        
        #=======================================================================
        # lookoup
        #=======================================================================
        dkey1 = 'fgdir_dxind'
        self.ofp_d[dkey1] = self.write_pick(fgdir_dxind, 
                           os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey1, self.longname)),logger=log)
        
        #save to data
        self.data_d[dkey1] = copy.deepcopy(fgdir_dxind)
        
        
        #=======================================================================
        # select result
        #=======================================================================
        if dkey == 'finv_agg':
            result = finv_agg_lib
        elif dkey == 'fgdir_dxind':
            result = fgdir_dxind

        
        return result
    """
    self.data_d.keys()
    """
    
    
    def load_finv_lib(self, #generic retrival for finv type intermediaries
                  fp=None, dkey=None,
                  **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('load_finv_lib.%s'%dkey)
        assert dkey in ['finv_agg',  'finv_gPoly', 'finv_sg_agg']
        
        
        vlay_fp_lib = self.load_pick(fp=fp) #{study area: aggLevel : vlay filepath}}
        
        #load layers
        finv_agg_lib = dict()
        
        for studyArea, vlay_fp_d in vlay_fp_lib.items():
            finv_agg_lib[studyArea] = dict()
            for aggLevel, fp in vlay_fp_d.items():
                log.info('loading %s.%s from %s'%(studyArea, aggLevel, fp))
                
                """will throw crs warning"""
                finv_agg_lib[studyArea][aggLevel] = self.vlay_load(fp, logger=log, **kwargs)
        
        return finv_agg_lib
 
 
                    
    

    def store_finv_lib(self, #consistent storage of finv containers 
                       finv_grid_lib, 
                       dkey,
                       out_dir = None,
                       logger=None):
        
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('store_finv')
        if out_dir is None: out_dir=os.path.join(self.wrk_dir, dkey)
        
        log.info('writing \'%s\' layers to %s' % (dkey, out_dir))
        
        #=======================================================================
        # write layers
        #=======================================================================
        ofp_d = dict()
        cnt = 0
        for studyArea, finv_grid_d in finv_grid_lib.items():
            #setup directory
            od = os.path.join(out_dir, studyArea)
            if not os.path.exists(od):
                os.makedirs(od)
            #write each sstudy area
            ofp_d[studyArea] = dict()
            for grid_size, poly_vlay in finv_grid_d.items():
                ofp_d[studyArea][grid_size] = self.vlay_write(poly_vlay, 
                    os.path.join(od, poly_vlay.name() + '.gpkg'), 
                    logger=log)
                cnt += 1
        
        log.debug('wrote %i layers' % cnt)
        #=======================================================================
        # filepahts container
        #=======================================================================
        
        #save the pickle
        """cant pickle vlays... so pickling the filepath"""
        self.ofp_d[dkey] = self.write_pick(ofp_d, 
            os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
        #save to data
        self.data_d[dkey] = finv_grid_lib
        return ofp_d

    def build_finv_gridPoly(self, #build polygon grids for each study area (and each grid_size)
                 dkey='finv_gPoly',
                 out_dir=None,
                 **kwargs):
        
        """this function constructs/stores two results:
            finv_gPoly: filepaths to merged grouped finvs {name:fp}
            fgdir_dxind: directory of finv keys from raw to grouped for each studyArea
            
        calling build on either will build both.. but only the result requsted will be returned
            (the other can be retrieved from data_d)
            """
        #=======================================================================
        # defaults
        #=======================================================================
        dkeys_l = ['finv_gPoly', 'finv_gPoly_id_dxind']
        log = self.logger.getChild('build_finv_gridPoly')
        
        if out_dir is None: out_dir=os.path.join(self.wrk_dir, dkey)
        
        #=======================================================================
        # prechecks
        #=======================================================================
        assert dkey in dkeys_l
        for dkey_chk in dkeys_l:
            if dkey_chk in self.data_d:
                log.warning('triggred reload on \'%s\''%dkey_chk)
                assert not dkey_chk == dkey, 'shouldnt reload any secondaries'
            
        
        #=======================================================================
        # #run the method on all the studyAreas
        #=======================================================================
        res_d = self.sa_get(meth='get_finvs_gridPoly', write=False, dkey=dkey, **kwargs)
        
        #unzip results
        finv_gkey_df_d, finv_grid_lib = dict(), dict() 
        for k,v in res_d.items():
            finv_gkey_df_d[k], finv_grid_lib[k]  = v
            
            #simple checks
            assert finv_gkey_df_d[k].columns.is_unique, 'got bad columns on \'%s\' \n    %s'%(
                k, finv_gkey_df_d[k].columns)
            
 
        #=======================================================================
        # handle layers----
        #=======================================================================
        dkey1 = 'finv_gPoly'
        self.store_finv_lib(finv_grid_lib, dkey1, out_dir = out_dir, logger=log)
        
        if dkey1 == dkey: result = finv_grid_lib
        
        #=======================================================================
        # handle dxcol-------
        #=======================================================================
        assert len(finv_gkey_df_d)>0
        
 
        
        dkey1 = 'finv_gPoly_id_dxind'
        dxind = pd.concat(finv_gkey_df_d)
        
        #check index
        assert dxind.columns.is_unique, 'bad columns on %s \n    %s'%(dkey1, dxind.columns)
        
        dxind.index.set_names('studyArea', level=0, inplace=True)
        dxind.columns.name = 'grid_size'
        self.check_mindex(dxind.index)

        #save the pickle
        self.ofp_d[dkey1] = self.write_pick(dxind, 
                           os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey1, self.longname)),logger=log)
        
        #save to data
        self.data_d[dkey1] = copy.deepcopy(dxind)
        
        if dkey1 == dkey: result = copy.deepcopy(dxind)
        #=======================================================================
        # return requested data
        #=======================================================================
        """while we build two results here... we need to return the one specified by the user
        the other can still be retrieved from the data_d"""
        
        
        
        return result
    

        

    def build_sampGeo(self, #get raster samples for all finvs
                     dkey='finv_sg_agg',
                     sgType = 'centroids',
 
                     **kwargs):
        
        #=======================================================================
        # defauts
        #=======================================================================
        assert dkey == 'finv_sg_agg'
        log = self.logger.getChild('build_sampGeo')
 
        
        finv_agg_lib = self.retrieve('finv_agg')
        
        #=======================================================================
        # loop each polygon layer and build sampling geometry
        #=======================================================================
        log.info('on %i w/ %s'%(len(finv_agg_lib), sgType))
        res_vlay_lib = dict()
        for studyArea, vlay_d in finv_agg_lib.items():
            res_vlay_lib[studyArea] = dict()
            for aggLevel, poly_vlay in vlay_d.items():
                log.info('on %s.%s w/ %i feats'%(studyArea, aggLevel, poly_vlay.dataProvider().featureCount()))
                
                if sgType == 'centroids':
                    """works on point layers"""
                    sg_vlay = self.centroids(poly_vlay, logger=log)
                    
                elif sgType == 'poly':
                    assert 'Polygon' in QgsWkbTypes().displayString(poly_vlay.wkbType()), 'bad type on %s.%s'%(studyArea, aggLevel)
                    poly_vlay.selectAll()
                    
                    sg_vlay = self.saveselectedfeatures(poly_vlay, logger=log) #just get a copy
                    
                else:
                    raise Error('not implemented')
                
                #===============================================================
                # wrap
                #===============================================================
                sg_vlay.setName('%s_%s'%(poly_vlay.name(), sgType))
                
                
                res_vlay_lib[studyArea][aggLevel] = sg_vlay
                
        
        #=======================================================================
        # store layers
        #=======================================================================
        ofp_d = self.store_finv_lib(res_vlay_lib, dkey,logger=log)
        
        return res_vlay_lib
                
        
 



    def build_rsamps(self, #get raster samples for all finvs
                     dkey=None,
                     method='points', #method for raster sampling
                     **kwargs):
        """
        keeping these as a dict because each studyArea/event is unique
        """
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey=='rsamps', dkey
        log = self.logger.getChild('build_rsamps')
 
        gcn=self.gcn
        #=======================================================================
        # child data
        #=======================================================================
        
        finv_aggS_lib = self.retrieve('finv_sg_agg')

        #=======================================================================
        # generate depths------
        #=======================================================================
        #=======================================================================
        # simple point-raster sampling
        #=======================================================================
        if method in ['points', 'zonal']:
            res_d = self.sa_get(meth='get_rsamps_lib', logger=log, dkey=dkey, write=False, 
                                finv_lib=finv_aggS_lib, idfn=gcn,method=method, **kwargs)
            
 
        

        #=======================================================================
        # use mean depths from true assets (for error isolation only)
        #=======================================================================
        elif method == 'true_mean':
            res_d = self.rsamp_trueMean(dkey,finv_aggS_lib,logger=log)
 
 
        #=======================================================================
        # shape into dxind
        #=======================================================================
        dxind1 = pd.concat(res_d, verify_integrity=True)
        dxind1.index.set_names('studyArea', level=0, inplace=True)
        
        """this will have some nulls as weve mashed all the events together
        assert dxind1.notna().all().all(), 'nulls should be replaced at lowset level'"""
        
 
        dxser = dxind1.stack(
            dropna=True, #zero values need to be set per-study area
            ).rename('depth') #promote depths to index
            
        dxser.index.set_names('event', level=3, inplace=True) 
        
        #=======================================================================
        # replace nulls with zeros
        #=======================================================================
        #$dxser = dxser.fillna(0.0)
        
        #=======================================================================
        # clean event names
        #=======================================================================
        #=======================================================================
        # """TODO: make a metatable for all the events then populate the names with something pretty and consistent"""
        # idf = dxser.index.to_frame().reset_index(drop=True)
        # idf['event'] = idf['event'].str.slice(start=5, stop=25).str.replace('_', '')
        # 
        # dxser.index = pd.MultiIndex.from_frame(idf)
        #=======================================================================
        #=======================================================================
        # #re-org
        #=======================================================================
        dxser = dxser.swaplevel().swaplevel(i=1,j=0).sort_index(axis=0, level=0, sort_remaining=True)
 
        
        #=======================================================================
        # checks
        #=======================================================================
        bx = dxser<0.0
        if bx.any().any():
            raise Error('got some negative depths')
        
        self.check_mindex(dxser.index)
 
 
        #=======================================================================
        # write
        #=======================================================================
        self.ofp_d[dkey] = self.write_pick(dxser, os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                        logger=log)
        
        return dxser
    
    
    def rsamp_trueMean(self, 
                       dkey, 
                       finv_aggS_lib,
                           logger=None,
                           ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_rsamp_trueMean')
        gcn = self.gcn
        #===================================================================
        # #get the true depths
        #===================================================================
        #just the true finvs
        finvT_lib = dict()
        for studyArea, lay_d in finv_aggS_lib.items():
            finvT_lib[studyArea] = {0:lay_d[0]}
        
        resT_d = self.sa_get(meth='get_rsamps_lib', logger=log, dkey=dkey, write=False, 
            finv_lib=finvT_lib, idfn=gcn)
 
        #===================================================================
        # get means
        #===================================================================
        fgdir_dxind = self.retrieve('fgdir_dxind')
        res_d = dict()
        for studyArea, lay_d in finv_aggS_lib.items():
            res_df = None
            for grid_size, vlay in lay_d.items():
                log.info('true_mean on %s.%s' % (studyArea, grid_size))
                rsampT_df = resT_d[studyArea].copy()
        #===========================================================
        # trues
        #===========================================================
                if grid_size == 0:
                    rdf = rsampT_df
                else:
                    id_gid_ser = fgdir_dxind.loc[idx[studyArea, :], grid_size].droplevel(0).rename(gcn)
                    rsampT_df.index.set_names(id_gid_ser.index.name, level=1, inplace=True)
                    rsampT_df1 = rsampT_df.join(id_gid_ser)
            #=======================================================
            # calc stat on gridded values
            #=======================================================
                    gsamp1_df = rsampT_df1.groupby(gcn).mean()
            #clean up index
                    gsamp1_df['grid_size'] = grid_size
                    rdf = gsamp1_df.set_index('grid_size', append=True).swaplevel()
            #=======================================================
            # check
            #=======================================================
                    fid_gid_d = vlay_get_fdata(vlay, fieldn=gcn)
            #gid_ser = pd.Series().sort_values().reset_index(drop=True)
                    miss_l = set(fid_gid_d.values()).difference(gsamp1_df.index)
                    assert len(miss_l) == 0 #=======================================================
        # #join gids to trues
        #=======================================================
                #===========================================================
                # join
                #===========================================================
        #===========================================================
        # gridded
        #===========================================================
                if res_df is None:
                    res_df = rdf
                else:
                    res_df = res_df.append(rdf)
            
            #===============================================================
            # wrap study area
            #===============================================================
            res_d[studyArea] = res_df
        log.info('got true_means on %i study areas'%len(res_d))
        return res_d
    
    #===========================================================================
    # HELPERS--------
    #===========================================================================
 
        

    def sa_get(self, #spatial tasks on each study area
                       proj_lib=None,
                       meth='get_rsamps', #method to run
                       dkey=None,
                       write=True,
                       logger=None,
                       **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('run_studyAreas')
        
        if proj_lib is None: proj_lib=self.proj_lib
        
        log.info('on %i \n    %s'%(len(proj_lib), list(proj_lib.keys())))
        
        
        #assert dkey in ['rsamps', 'finv_gPoly'], 'bad dkey %s'%dkey
        
        #=======================================================================
        # loop and load
        #=======================================================================
        res_d = dict()
        for i, (name, pars_d) in enumerate(proj_lib.items()):
            log.info('%i/%i on %s'%(i+1, len(proj_lib), name))
            
            with StudyArea(session=self, name=name, tag=self.tag, prec=self.prec,
                           trim=self.trim, out_dir=self.out_dir, overwrite=self.overwrite,
                           **pars_d) as wrkr:
                
                #raw raster samples
                f = getattr(wrkr, meth)
                res_d[name] = f(**kwargs)
                
        #=======================================================================
        # write
        #=======================================================================
        if write:
            """frames between study areas dont have any relation to each other... keeping as dict"""

            self.ofp_d[dkey] = self.write_pick(res_d, os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)), logger=log)
 
        
        return res_d
    
    def check_mindex(self, #check names and types
                     mindex,
                     chk_d = None,
                     logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        #if logger is None: logger=self.logger
        #log=logger.getChild('check_mindex')
        if chk_d is None: chk_d=self.mindex_dtypes
        
        #=======================================================================
        # check types and names
        #=======================================================================
        names_d= {lvlName:i for i, lvlName in enumerate(mindex.names)}
        
        assert not None in names_d, 'got unlabled name'
        
        for name, lvl in names_d.items():
 
            assert name in chk_d, 'name \'%s\' not recognized'%name
            assert mindex.get_level_values(lvl).dtype == chk_d[name], \
                'got bad type on \'%s\': %s'%(name, mindex.get_level_values(lvl).dtype.name)
                
        #=======================================================================
        # check index values
        #=======================================================================
        #totality is unique
        bx = mindex.to_frame().duplicated()
        assert not bx.any(), 'got %i/%i dumplicated index entries on %i levels \n    %s'%(
            bx.sum(), len(bx), len(names_d), names_d)
        
 
        
                
        
        return
            
        
 
 

        
class StudyArea(Session, Qproj): #spatial work on study areas
    
    
    finv_fnl = [] #allowable fieldnames for the finv
    def __init__(self, 
                 
                 #pars_lib kwargs
                 EPSG=None,
                 finv_fp=None,
                 dem=None,
                 wd_dir=None,
                 aoi=None,
                 
                 #control
                 trim=True,
                 idfn = 'id', #unique key for assets
                 **kwargs):
        
        super().__init__(**kwargs)
        
        #=======================================================================
        # #set crs
        #=======================================================================
        crs = QgsCoordinateReferenceSystem('EPSG:%i'%EPSG)
        assert crs.isValid()
            
        self.qproj.setCrs(crs)
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.idfn=idfn
        
        self.finv_fnl.append(idfn)
        #=======================================================================
        # load aoi
        #=======================================================================
        if not aoi is None and trim:
            self.load_aoi(aoi)
            
        #=======================================================================
        # load finv
        #=======================================================================
        finv_raw = self.vlay_load(finv_fp, dropZ=True, reproj=True)
        
        #field slice
        fnl = [f.name() for f in finv_raw.fields()]
        drop_l =  list(set(fnl).difference(self.finv_fnl))
        if len(drop_l)>0:
            finv1 = self.deletecolumn(finv_raw, drop_l)
            self.mstore.addMapLayer(finv_raw)
        else:
            finv1 = finv_raw
        
        #spatial slice
        if not self.aoi_vlay is None:
            finv2 = self.slice_aoi(finv1)
            self.mstore.addMapLayer(finv1)
        
        else:
            finv2 = finv1
        
        finv2.setName(finv_raw.name())
 
            
        #check
        miss_l = set(self.finv_fnl).symmetric_difference([f.name() for f in finv2.fields()])
        assert len(miss_l)==0, 'unexpected fieldnames on \'%s\' :\n %s'%(miss_l, finv2.name())
 
  
        #=======================================================================
        # attachments
        #=======================================================================
        self.finv_vlay = finv2
        self.wd_dir=wd_dir
 
    def get_clean_rasterName(self, raster_fn,
                             conv_lib = {
                                'LMFRA':{
                                    'AG4_Fr_0050_dep_0116_cmp.tif':'0050yr',
                                    'AG4_Fr_0100_dep_0116_cmp.tif':'0100yr',
                                    'AG4_Fr_0500_dep_0116_cmp.tif':'0500yr',
                                    'AG4_Fr_1000_dep_0116_cmp.tif':'1000yr',
                                        },
    
                                        }   
                             ):
        
        rname = raster_fn.replace('.tif', '')
        if self.name in conv_lib:
            if raster_fn in conv_lib[self.name]:
                rname = conv_lib[self.name][raster_fn]
                
        return rname
        
        
        
    def get_finvs_gridPoly(self, #get a set of polygon grid finvs (for each grid_size)
                  finv_vlay_raw=None,
                  grid_sizes = [5, 20, 100], #resolution in meters
                  idfn=None,
 
                  overwrite=None,
                  ):
        """
        
        how do we store an intermitten here?
            study area generates a single layer on local EPSG
            
            Session writes these to a directory

        need a meta table
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_finvs_gridPoly')
        gcn = self.gcn
        if overwrite is None: overwrite=self.overwrite
        if finv_vlay_raw is None: finv_vlay_raw=self.finv_vlay
        if idfn is None: idfn=self.idfn
        
        #=======================================================================
        # clean finv
        #=======================================================================
        """general pre-cleaning of the finv happens in __init__"""
        
        drop_fnl = set([f.name() for f in finv_vlay_raw.fields()]).difference([idfn])
 
        if len(drop_fnl)>0:
            finv_vlay = self.deletecolumn(finv_vlay_raw, list(drop_fnl), logger=log)
            self.mstore.addMapLayer(finv_vlay) #keep the raw alive
        else:
            finv_vlay = finv_vlay_raw
 
        #=======================================================================
        # loop and aggregate
        #=======================================================================
        finv_grid_d = dict() #container for resulting finv points layers
        groups_d = dict()
        meta_d = dict()
 
        log.info('on \'%s\' w/ %i: %s'%(finv_vlay.name(), len(grid_sizes), grid_sizes))
        
        for i, grid_size in enumerate(grid_sizes):
            log.info('%i/%i grid w/ %.1f'%(i+1, len(grid_sizes), grid_size))
            
            #===================================================================
            # raw grid
            #===================================================================
            
            gvlay1 = self.creategrid(finv_vlay, spacing=grid_size, logger=log)
            self.mstore.addMapLayer(gvlay1)
            log.info('    built grid w/ %i'%gvlay1.dataProvider().featureCount())
            
 
            #===================================================================
            # active grid cells only
            #===================================================================

            
            #select those w/ some assets
            gvlay1.removeSelection()
            self.createspatialindex(gvlay1)
            log.info('    selecting from grid based on intersect w/ \'%s\''%(finv_vlay.name()))
            self.selectbylocation(gvlay1, finv_vlay, logger=log)
            
            #save these
            gvlay2 = self.saveselectedfeatures(gvlay1, logger=log, output='TEMPORARY_OUTPUT')
 
            #===================================================================
            # populate/clean fields            
            #===================================================================
            #rename id field
            gvlay3 = self.renameField(gvlay2, 'id', gcn, logger=log)
            self.mstore.addMapLayer(gvlay3)
            log.info('    renamed field \'id\':\'%s\''%gcn)
            
            
            #delete grid dimension fields
            fnl = set([f.name() for f in gvlay3.fields()]).difference([gcn])
            gvlay3b = self.deletecolumn(gvlay3, list(fnl),  logger=log)
            self.mstore.addMapLayer(gvlay3b)
            
            #add the grid_size
            gvlay4 = self.fieldcalculator(gvlay3b, grid_size, fieldName='grid_size', 
                                           fieldType='Integer', logger=log)
            
 
            
            #===================================================================
            # build refecrence dictionary to true assets
            #===================================================================
            jd = self.joinattributesbylocation(finv_vlay, gvlay4, jvlay_fnl=gcn, 
                                               method=1, logger=log,
                                               #predicate='touches',
                     output_nom=os.path.join(self.temp_dir, 'finv_grid_noMatch_%i_%s.gpkg'%(
                                                 grid_size, self.longname)))
            
            #check match
            noMatch_cnt = finv_vlay.dataProvider().featureCount() - jd['JOINED_COUNT']
            if not noMatch_cnt==0:
                """gid lookup wont work"""
                raise Error('for \'%s\' grid_size=%i failed to join  %i/%i assets... wrote non matcherse to \n    %s'%(
                    self.name, grid_size, noMatch_cnt, finv_vlay.dataProvider().featureCount(), jd['NON_MATCHING']))
                    
            jvlay = jd['OUTPUT']
            self.mstore.addMapLayer(jvlay)
            
            df = vlay_get_fdf(jvlay, logger=log).set_index(idfn)
            
            #===================================================================
            # check against grid points
            #===================================================================
            """
            this is an artifact of doing selectbylocation then joinattributesbylocation
                sometimes 1 or 2 grid cells are erroneously joined
                here we just delete them
            """
            gpts_ser = vlay_get_fdf(gvlay4)[gcn]
            
            set_d = set_info(gpts_ser.values, df[gcn].values)

            
            if not len(set_d['symmetric_difference'])==0:
                del set_d['union']
                del set_d['intersection']
                log.warning('%s.%i got %i mismatched values... deleteing these grid cells\n   %s'%(
                    self.name, grid_size, len(set_d['symmetric_difference']), set_d))
                
                assert len(set_d['diff_right'])==0
                
                #get matching ids
                fid_l = gpts_ser.index[gpts_ser.isin(set_d['diff_left'])].tolist()
                gvlay4.removeSelection()
                gvlay4.selectByIds(fid_l)
                assert gvlay4.selectedFeatureCount()==len(set_d['diff_left'])
                
                #delete these
                gvlay4.invertSelection()
                gvlay4 = self.saveselectedfeatures(gvlay4, logger=log)
                
            #===================================================================
            # write
            #===================================================================
            
            gvlay4.setName('finv_gPoly_%i_%s'%(grid_size, self.longname.replace('_', '')))
            """moved onto the session
            if write_grids:
                self.vlay_write(gvlay4, os.path.join(od, gvlay4.name() + '.gpkg'),
                                logger=log)"""
 
            #===================================================================
            # wrap
            #===================================================================
            finv_grid_d[grid_size]=gvlay4
            groups_d[grid_size] = df.copy()
 
            #===================================================================
            # #meta
            #===================================================================
            mcnt_ser = df[gcn].groupby(df[gcn]).count()
            meta_d[grid_size] = {
                'total_cells':gvlay1.dataProvider().featureCount(), 
                'active_cells':gvlay1.selectedFeatureCount(),
                'max_member_cnt':mcnt_ser.max()
                }
            
 
            log.info('    joined w/ %s'%meta_d[grid_size])
            
        #=======================================================================
        # add trues
        #=======================================================================
 
        grid_size = 0
        if not 'Polygon' in QgsWkbTypes().displayString(finv_vlay.wkbType()):
            log.warning('mixed types in finv_lib')
            
            """consider using a common function for this and the above"""
            
            
        #rename the id field
        tgvlay1 = self.renameField(finv_vlay, idfn, gcn, logger=log)
        self.mstore.addMapLayer(tgvlay1)
        
        #remove other fields
        fnl = set([f.name() for f in tgvlay1.fields()]).difference([gcn])
        assert len(fnl)==0
        #tgvlay2 = self.deletecolumn(tgvlay1, list(fnl),  logger=log)
        #self.mstore.addMapLayer(tgvlay2)
        
        
        #add the grid_size
        tgvlay3 = self.fieldcalculator(tgvlay1, grid_size, fieldName='grid_size', 
                                       fieldType='Integer', logger=log)
        
        
        tgvlay3.setName('finv_gPoly_%i_%s'%(grid_size, self.longname.replace('_', '')))
        
        finv_grid_d[grid_size]=tgvlay3
        
        meta_d[grid_size] = {
            'total_cells':tgvlay3.dataProvider().featureCount(), 
            'active_cells':tgvlay3.dataProvider().featureCount(),
            'max_member_cnt':1
            }
 
        #=======================================================================
        # wrap
        #=======================================================================
        
        log.info('finished on %i'%len(groups_d))
 
        #assemble the grid id per raw asset
        finv_gkey_df = pd.concat(groups_d, axis=1).droplevel(axis=1, level=1)
        
        #=======================================================================
        # #checks
        #=======================================================================
        miss_l = set(finv_gkey_df.columns).symmetric_difference(grid_sizes)
        assert len(miss_l)==0, 'grid_sizes dont match on output'
        
        assert finv_gkey_df.columns.is_unique
        
 
        
        return finv_gkey_df, finv_grid_d
    

        
 
    def get_rsamps_lib(self, #get samples for a set of layers
                       finv_lib=None,
 
                       **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_rsamps_lib')
 
        
        #=======================================================================
        # retrieve
        #=======================================================================
        #pull your layers from the session
        finv_vlay_d = finv_lib[self.name]
        
        
        #=======================================================================
        # loop and sample each
        #=======================================================================
        log.info('sampling on %i finvs'%len(finv_vlay_d))
        
        res_d = dict()
        for aggLevel, finv_vlay in finv_vlay_d.items():
            res_d[aggLevel] = self.get_rsamps(finv_vlay_raw=finv_vlay, logger=self.logger.getChild('aL%i'%aggLevel),
                            **kwargs)
            
        #=======================================================================
        # combine
        #=======================================================================
        dxind = pd.concat(res_d, verify_integrity=True).sort_index()
        dxind.index.set_names('grid_size', level=0, inplace=True)
        
        """
        view(dxind)
        """
        #check
        self.session.check_mindex(dxind.index)
        assert dxind.notna().all().all(), 'zeros replaced at lowset level'
        
        dry_bx = dxind<=0
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i vlays and %i rasters w/ %i/%i dry'%(
            len(finv_vlay_d), len(dxind.columns), dry_bx.sum().sum(), dxind.size))
        
 
        
        return dxind
                       
        



    def get_rsamps(self, #sample a set of rastsers withon a single finv
                   wd_dir = None,
                   finv_vlay_raw=None,
                   idfn=None,
                   logger=None,
                   method='points',
                   zonal_stats = [2], #stats to use for zonal. 2=mean
                   prec=None,
                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_rsamps')
        if wd_dir is None: wd_dir = self.wd_dir
        if finv_vlay_raw is None: finv_vlay_raw=self.finv_vlay
        if idfn is None: idfn=self.idfn
        if prec is None: prec=self.prec
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert os.path.exists(wd_dir)
        if method=='points':
            assert 'Point' in QgsWkbTypes().displayString(finv_vlay_raw.wkbType())
        elif method=='zonal':
            assert 'Polygon' in QgsWkbTypes().displayString(finv_vlay_raw.wkbType())
            assert isinstance(zonal_stats ,list)
            assert len(zonal_stats)==1
            
            
        #=======================================================================
        # clean finv
        #=======================================================================
        """general pre-cleaning of the finv happens in __init__"""
        
        drop_fnl = set([f.name() for f in finv_vlay_raw.fields()]).difference([idfn])
 
        if len(drop_fnl)>0:
            finv_vlay = self.deletecolumn(finv_vlay_raw, list(drop_fnl), logger=log)
            self.mstore.addMapLayer(finv_vlay) #keep the raw alive
        else:
            finv_vlay = finv_vlay_raw
            
            
        #=======================================================================
        # retrieve
        #=======================================================================
        fns = [e for e in os.listdir(wd_dir) if e.endswith('.tif')]
        
        log.info('on %i rlays \n    %s'%(len(fns), fns))
        
        #=======================================================================
        # loop and sample
        #=======================================================================
        res_df = None
        for i, fn in enumerate(fns):
            rname = self.get_clean_rasterName(fn)
            log.debug('%i %s'%(i, fn))
            
            rlay_fp = os.path.join(wd_dir, fn)
            
            
            #===================================================================
            # sample
            #===================================================================
            if method=='points':
                vlay_samps = self.rastersampling(finv_vlay, rlay_fp, logger=log, pfx='samp_')
 
 
            
            elif method=='zonal':
                vlay_samps = self.zonalstatistics(finv_vlay, rlay_fp, logger=log, pfx='samp_', stats=zonal_stats)
                

            #===================================================================
            # post           
            #===================================================================
            self.mstore.addMapLayer(vlay_samps)
            #change column names
            df = vlay_get_fdf(vlay_samps, logger=log)
            df = df.rename(columns={df.columns[1]:rname})
            
            
            
            #force type
            assert idfn in df.columns, 'missing key \'%s\' on %s'%(idfn, finv_vlay.name())
            df.loc[:, idfn] = df[idfn].astype(np.int64)
            
            assert df[idfn].is_unique, finv_vlay.name()

            
            rdf = df.set_index(idfn).drop('grid_size', axis=1, errors='ignore')
            if res_df is None:
                res_df = rdf
            else:
                res_df = res_df.join(rdf, how='outer')
            
 
        
        #=======================================================================
        # fill zeros
        #=======================================================================
        res_df1 = res_df.fillna(0.0)
        #=======================================================================
        # wrap
        #=======================================================================

        assert res_df.index.is_unique
        
        
        log.info('finished on %s and %i rasters w/ %i/%i dry'%(
            finv_vlay.name(), len(fns), res_df.isna().sum().sum(), res_df.size))

        
        return res_df1.round(prec).astype(np.float32).sort_index()
    
 
 
        