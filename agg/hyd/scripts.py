'''
Created on Jan. 18, 2022

@author: cefect



'''
import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np
idx = pd.IndexSlice



from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo, \
    QgsWkbTypes

 
from hp.basic import set_info

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from agg.scripts import Session as agSession

import matplotlib


class Session(agSession):
    
    gcn = 'gid'
    
    mindex_dtypes={
                 'studyArea':np.dtype('object'),
                 'id':np.dtype('int64'),
                 'gid':np.dtype('int64'), #both ids are valid
                 'grid_size':np.dtype('int64'), 
                 'event':np.dtype('O'),           
                         }
    
    colorMap = 'cool'
    
    def __init__(self, 
                 name='hyd',
                 proj_lib={},
                 trim=True, #whether to apply aois
                 **kwargs):
        
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
            
            'rsamps':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_rsamps(**kwargs),
                },
            
            'rloss':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_rloss(**kwargs),
                },
            'tloss':{
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
                    plot_fign = 'studyArea',
                    plot_rown = 'grid_size', 
                    plot_coln = 'event',
 
                    
                    plot_zeros=False,
                    xlims = (0,2),
                    ylims=(0,2.5),
                    
                    out_dir=None,
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_depths')
        if out_dir is None: out_dir = os.path.join(self.out_dir, 'depths')
        #=======================================================================
        # #retrieve child data
        #=======================================================================
 
        serx = self.retrieve('rsamps')
        
        
        
        assert serx.notna().all().all(), 'drys should be zeros'

        """
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
                
                #label
                meta_d = {
                    plot_rown:row_key,
                    'wet':len(ar), 'dry':(gsx2<=0.0).sum(), 'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
                
                txt = '\n'.join(['%s=%.2f'%(k,v) for k,v in meta_d.items()])
                ax.text(0.5, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='blue')
                

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
            res_d[sName]= self.output_fig(fig, out_dir=out_dir, fname='depths_%s_%s'%(sName, self.longname))

        #=======================================================================
        # warp
        #=======================================================================
        log.info('finished writing %i figures'%len(res_d))
        
        return res_d
                    
 
            
            
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
                # add labels
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
                        ax.text(row['xloc'], row['pred'], '%.1f'%(row['relErr']*100),
                                ha='center', va='bottom', rotation='vertical',
                                fontsize=8,color='red')
                        
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
                    lossType = 'tl', 
                    ycoln = 'delta', #variable of the box plots
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
        if out_dir is None: out_dir=os.path.join(self.out_dir, 'errs')
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
        
        view(dx_raw)
        
        """
 
         
        #get the yvariable to includein the box plot
        dxind1 = dx_raw.loc[:, idx[lossType, ycoln, :]].droplevel(level=[0, 1], axis=1)
 
        dxind1.columns = dxind1.columns.astype(int)

        #promote column values to index
        dxser = dxind1.stack().rename(ycoln)
 
        
        #=======================================================================
        # plotting setup
        #=======================================================================
        #get colors
        #=======================================================================
        # cvals = dxser.index.unique(plot_colr)
        # cmap = plt.cm.get_cmap(name=colorMap) 
        # newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        #=======================================================================
        
        plt.close('all')
        """
        plt.show()
        """
        #=======================================================================
        # loop on figures
        #=======================================================================
        for i, (fig_key, gser0r) in enumerate(dxser.groupby(level=plot_fign)):
            mdex = gser0r.index
            
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
            
            fig.suptitle('%s of %s for %s \'%s\' (%s)'%(ycoln, lossType.upper(), plot_fign, fig_key, self.tag))
        

        
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
            self.output_fig(fig, fname='box_%s_%s_%s_%s'%(lossType, ycoln, fig_key, self.longname), 
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
                    ycoln = idx['rl', :, 'delta'],
                    #xcoln = idx[,],
                       
                    #figure config
                    folder_varn = 'studyArea',
                    plot_fign = 'event',
                    plot_rown = 'grid_size', 
                    plot_coln = 'vid',
                    plot_colr = None,
                    #plot_bgrp = 'event',
                    out_dir = None,
                    
                    plot_vf=False, #plot the vf
                    plot_zeros=False,
                    
                    #axconfig
                    ylims = (-2,2),
                    xlims = (0,3),
                    
                    #plot style
                    ylabel=None,
                    colorMap=None,
                    add_text=True,
                   ):
        
        #raise Error('lets fit a regression to these results')
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_errs_scatter')
        if colorMap is None: colorMap=self.colorMap
        #if ylabel is None: ylabel = '%s of %s'%(ycoln, lossType.upper())
        
        #=======================================================================
        # if plot_vf:
        #     assert lossType=='rl'
        #=======================================================================
            
        if plot_colr is None: plot_colr=plot_rown
        if out_dir is None: out_dir = os.path.join(self.out_dir, 'errs')
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
 
        if plot_vf:
            vf_d = self.retrieve('vf_d')
        log.info('on \'%s\' w/ %i'%(dkey, len(dx_raw)))
        
        #=======================================================================
        # setup data
        #=======================================================================
        """
        moving everything into an index for easier manipulation
        view(dx_raw)
        """
 
        
        dxind1 = dx_raw.loc[:, idx[lossType, ycoln, :]].droplevel([0, 1], axis=1)

        #promote column values to index
        dxser = dxind1.stack().rename(ycoln).sort_index(sort_remaining=True)
 
        
        #get yvalues (depths)
        
        xval_dxser = self.retrieve('rsamps').copy().sort_index(sort_remaining=True)
        assert xval_dxser.name == xcoln
        #=======================================================================
        # plotting setup
        #=======================================================================
        #get colors
        cvals = dxser.index.unique(plot_colr)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        
        """
        plt.show()
        """
        #=======================================================================
        # loop an study area/folders
        #=======================================================================
        
        for folder_key, gser00r in dxser.groupby(level=folder_varn):
            
            #=======================================================================
            # loop on figures
            #=======================================================================
            od = os.path.join(out_dir, folder_key)
            plt.close('all')
        
            for i, (fig_key, gser0r) in enumerate(gser00r.groupby(level=plot_fign)):
                mdex = gser0r.index
                
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
                fig.suptitle('%s of %s for %s \'%s\' in \'%s\''%(ycoln, lossType.upper(), plot_fign, fig_key, folder_key))
            
                """
                fig.show()
                """
            
                #===================================================================
                # loop on axis row/column (and colors)----------
                #===================================================================
                for (row_key, col_key, ckey), gser1r in gser0r.groupby(level=[plot_rown, plot_coln, plot_colr]):
                    keys_d = dict(zip([plot_rown, plot_coln, plot_fign], (row_key, col_key, fig_key)))
                    #skip trues
                    #===========================================================
                    # if ckey == 0:
                    #     continue 
                    #===========================================================
                    #subplot setup 
                    ax = ax_d[row_key][col_key]
                    
     
                    #===============================================================
                    # #get data
                    #===============================================================
                    assert gser1r.notna().all().all()
                    
                    #cross section of depth values
                    xser = xval_dxser.xs((fig_key, row_key, ckey), level=[plot_fign, plot_rown, plot_colr])
                    
                    #check
                    miss_l = set(xser.index).symmetric_difference(gser1r.index.unique(xser.index.name))
                    assert len(miss_l)==0, 'index mismatch w/ depths on %s'%(keys_d)
                    
                    #join to xvalues
                    """zeros are dry"""
                    dxind = gser1r.to_frame().join(xser, on=xser.index.name)
                    
                    dry_bx = dxind['depth']<=0.0
                    
                    if not plot_zeros:
                        xar, yar = dxind.loc[~dry_bx, xcoln], dxind.loc[~dry_bx, ycoln]
                    else:
                        xar, yar = dxind[xcoln], dxind[ycoln]

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
                                  'mean':yar.mean(), 'min':yar.min(), 'max':yar.max(),
                              #'plot_zeros':plot_zeros,
                              }
                        
                        if ycoln == 'delta':
                            meta_d['rmse'] = ((gser1r.values**2).mean())**0.5
                                            
                        txt = '\n'.join(['%s=%.2f'%(k,v) for k,v in meta_d.items()])
                        ax.text(0.1, 0.8, txt, transform=ax.transAxes, va='top', fontsize=8, color='black')
     
     
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
                        ax.set_xlabel(xcoln)
                        
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
                self.output_fig(fig, out_dir=od, 
                                fname='scatter_%s_%s_%s_%s'%(lossType.upper(),ycoln, fig_key, self.longname),
                                fmt='svg', logger=log)
            """
            fig.show()
            """
                    

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
    
    
    def plot_accuracy_mat(self,
                    #data control   
                    dkey='errs',
                    lossType='rl',
 
                    
                    folder_keys = ['studyArea', 'event'], 
                    plot_fign = 'vid', #one raster:vid per plot
                    plot_rown = 'grid_size', 

                    #output control
                    out_dir = None,
                    
 
                    #axconfig
                    ylims = (-2,2),
                    xlims = (0,3),
                    
                    #plot style
 
                    colorMap=None,
                    add_text=True,
                   ):
        
        """would be nice if this could plot
            rl
            tl
            depths
            """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_errs_scatter')
        if colorMap is None: colorMap=self.colorMap
 
        if out_dir is None: out_dir = os.path.join(self.out_dir, 'errs')
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        """
        view(dx_raw)
        """
        
        

    
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
        view(dxind1)
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
            if lossType == 'meta':continue
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
        dindex2 = dx_raw.loc[:, idx['meta', :]].droplevel(0, axis=1)
        
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
        gbo = dindex2['id_cnt'].groupby(level=gkeys)
        
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
                    
                    lossType='rl', #loss types to generate layers for
                    
                    #output config
                    folder1_key = 'studyArea',
                    folder2_key = 'event',
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
        if out_dir is None:
            out_dir = os.path.join(self.out_dir, 'errs')
            
        gcn = self.gcn
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        #errors
        dx_raw = self.retrieve(dkey)
        mindex = dx_raw.index
        names_d= {lvlName:i for i, lvlName in enumerate(dx_raw.index.names)}
        
        #get type
        dx1 = dx_raw.loc[:, idx[lossType, :,:,:]].droplevel(0, axis=1)
        
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
        log.info('on \'%s\' w/ %i'%(dkey, len(dx1)))
        
        lvls = [names_d[k] for k in [folder1_key, folder2_key, file_key]]
        for i, (keys, gdx) in enumerate(dx1.groupby(level=lvls)):
            keys_d = dict(zip([mindex.names[i] for i in lvls], keys))
            log.debug(keys_d)
            

                
            #===================================================================
            # retrieve spatial data
            #===================================================================
            #get vlay
            finv_vlay = finv_agg_lib[keys_d['studyArea']][keys_d['grid_size']]
             
            
#===============================================================================
#             #get key dictionary off layer
#             fgm_df = vlay_get_fdf(fgm_vlay, logger=log)
#             
#             fgm_ser1 = fgm_df.loc[fgm_df['grid_size']==keys_d['grid_size'], ['fid', 'gid']].set_index('fid').iloc[:,0]
#             
#             #check key match
#             miss_l = set(gdx.index.unique('gid')).difference(fgm_ser1.values)
#             assert len(miss_l)==0, 'missing %i entries found in dxcol but not in fgm_vlay: \n    %s'%(len(miss_l), miss_l)
#             
#             #get fid match
#             gf_all_d = {v:k for k,v in fgm_ser1.to_dict().items()} #all the keys
#             gf_d = {gid:gf_all_d[gid] for gid in gdx.index.unique('gid')} #just those in the index
#             
# 
#             #get geometries
#             request = QgsFeatureRequest().setFilterFids(list(gf_d.values()))
#             geo_d = vlay_get_geo(fgm_vlay, request=request)
#===============================================================================
            
            geo_d = vlay_get_geo(finv_vlay)
            fid_gid_d = vlay_get_fdata(finv_vlay, fieldn=gcn)
            #===================================================================
            # prepare data
            #===================================================================
            #compress columns
            cdf = gdx.columns.to_frame().reset_index(drop=True)
            
            gdxind = gdx.copy()
            gdxind.columns = cdf.iloc[:,0].str.cat(cdf.iloc[:,1].astype(str), sep='_')
            
            #get depths and i_cnt
            bx = tl_dxind.index.isin(gdx.index)
            mdxind = tl_dxind.loc[bx, idx['meta', :]].droplevel(0, axis=1)
            gdxind = gdxind.join(mdxind, on=gdx.index.names)
            
            #get flat frame
            gdf1 = pd.DataFrame(gdxind.values,
                index=gdx.index.droplevel(lvls),
                columns=gdxind.columns).reset_index()
            
 
            #reset index
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
            od = os.path.join(out_dir, keys_d[folder1_key], keys_d[folder2_key])
            if not os.path.exists(od):
                os.makedirs(od)
                
            ofp = os.path.join(od, self.longname + '_' + lossType + '_' + layname  + '.gpkg')
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
                             dkey = 'errs',
                             group_keys = ['studyArea', 'event'],
                             pcn = 'grid_size', #label for prediction grouping
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
        vid = dx_raw.columns.unique('vid')[0] #just taking one function
        dx1 = dx_raw.loc[:, idx['rl', ['grid', 'true'],vid]].droplevel([0, 2], axis=1)
        
        #replace all 'grided' on grid_size zero
        dx1.loc[idx[0, :,:], 'grid'] = dx1.loc[idx[0, :,:], 'true']
        assert dx1.notna().all().all()
 
        """
        view(dx_raw)
        view(dx1)
        view(dx2)
        """
 
        # convert to binary
        bx = dx1>0 #id wets
        
        dx2 = dx1.where(~bx, other=labs_d[1]).where(bx, other=labs_d[0]) #replace where false
        
        #=======================================================================
        # loop on each raster
        #=======================================================================
        mat_lib = dict()
        for keys, gdx0 in dx2.groupby(level=group_keys):
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
            #===================================================================
            # disp = ConfusionMatrixDisplay(confusion_matrix=dx2.values)
            # 
            # disp.plot()
            # 
            # ax = disp.ax_
            # 
            # ax.set_title(keys_d)
            #===================================================================
            
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
                elif ckeys_d['vid'] == 'id_cnt':    
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
        """
        view(res_dx.drop(0, level='grid_size', axis=0))
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
 
        
        self.ofp_d[dkey] = self.write_pick(res_dx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                                   logger=log)

        return res_dx
 
 
    
    
    def build_tloss(self, #get the total loss
                    dkey=None,
                    prec=2,
                    ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_tloss')
        assert dkey=='tloss'
        if prec is None: prec=self.prec
        gcn = self.gcn
        scale_cn = 'id_cnt'
        #=======================================================================
        # retriever
        #=======================================================================
        rl_dxind = self.retrieve('rloss')
        
        rlnames_d= {lvlName:i for i, lvlName in enumerate(rl_dxind.index.names)}
        
        fgdir_dxind = self.retrieve('fgdir_dxind')
        
        """
        view(rl_dxind)
        view(fgdir_dxind)
        view(ser1)
        """
        #=======================================================================
        # calculate the asset scalers
        #=======================================================================
        grid_d =dict()
        for grid_size, gdf in rl_dxind.groupby(level=rlnames_d['grid_size']):
            #get teh scaler series
            if grid_size==0:
                #unique studyArea:gid
                mindex = gdf.index.droplevel(['grid_size', 'event']).drop_duplicates()
                cnt_serix = pd.Series(1, index=mindex)
                                      
 
            else:
                #retrieve the lookup studyArea+id:gid
                ser1 = fgdir_dxind.loc[:, grid_size].rename(gcn)
                
                assert ser1.notna().all(), grid_size
                
                
                #get counts per-area (for this group size)
                d = dict()
                for sName, ser2 in ser1.groupby(level=0):                
                    d[sName] = ser2.groupby(ser2).count()
 
                cnt_serix = pd.concat(d, names=mindex.names)
                
            #===================================================================
            # #checks
            #===================================================================
            assert cnt_serix.notna().all(), grid_size
            try:
                self.check_mindex(cnt_serix.index)
            except Exception as e:
                raise Error('index on grid_size=%i failed w/ \n    %s'%(grid_size, e))
            
            #all the keys needed by the right are in the left
            miss_l = set(gdf.index.unique(gcn)).difference(cnt_serix.index.unique(gcn))
            if not len(miss_l)==0:
                bx = rl_dxind.index.to_frame().loc[:,gcn].isin(miss_l)
 
                
                with pd.option_context('display.max_rows', None,'display.max_columns', None,'display.width',1000):
                    log.debug('\n%s'%rl_dxind.loc[bx,:].index.to_frame().reset_index(drop=True))
        
 
                raise Error('missing %i lookup \'%s\' values... see logger'%(len(miss_l), gcn))
            
            #wrap
            grid_d[grid_size] = cnt_serix.rename(scale_cn)

        
        #shape into a dxind with matching names
        jserx1 = pd.concat(grid_d, names= rl_dxind.droplevel(rlnames_d['event']).index.names)

        
        
        #=======================================================================
        # check
        #=======================================================================
        self.check_mindex(jserx1.index)
 
        
        jnames_d = {lvlName:i for i, lvlName in enumerate(jserx1.index.names)}
        assert jserx1.notna().all()
        
        #check index
        """we expect more in the scaleing lookup (jserx1) as these have not been filtered for zero impacts"""
        for name in jnames_d.keys():
            
            left_vals = rl_dxind.index.get_level_values(rlnames_d[name]).unique()
            right_vals = jserx1.index.get_level_values(jnames_d[name]).unique()
            
            el_d = set_info(left_vals, right_vals)
            
            
            
            for k,v in el_d.items():
                log.debug('%s    %s'%(k,v))
                
            if not len(el_d['diff_left'])==0:
                
                bx = rl_dxind.index.to_frame().loc[:,name].isin(el_d['diff_left'])
                assert bx.sum() == len(el_d['diff_left'])
                
                with pd.option_context('display.max_rows', None,'display.max_columns', None,'display.width',1000):
                    log.debug(rl_dxind.loc[bx,:].index.to_frame().reset_index(drop=True))
        
 
                raise Error('%s missing %i some lookup values'%(name, len(el_d['diff_left'])))
        
 
        """
        I think we get very few on grid_size=0 because this as been null filtered
        view(jdxind1)
        """
        
        #=======================================================================
        # #join to the rloss data
        #=======================================================================
        dxind1 = rl_dxind.join(jserx1, on=jserx1.index.names)
        
        assert dxind1[scale_cn].notna().all()
        """
        view(dxind1)
        """
        
        #re-org columns
        
        dx1 = pd.concat({
            'expo':dxind1.loc[:, ['depth', scale_cn]],
            'rl':dxind1.loc[:, rl_dxind.drop('depth', axis=1).columns]},
            axis=1)
        
 
        
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

    
    def build_rloss(self,
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
        #finv_gkey_df, finv_grid_d
        finv_gkey_df_d, finv_grid_lib = dict(), dict() 
        for k,v in res_d.items():
            finv_gkey_df_d[k], finv_grid_lib[k]  = v
            
 
        #=======================================================================
        # handle layers----
        #=======================================================================
        dkey1 = 'finv_gPoly'
        self.store_finv_lib(finv_grid_lib, dkey1, out_dir = out_dir, logger=log)
        
        if dkey1 == dkey: result = finv_grid_lib
        
        #=======================================================================
        # handle dxcol-------
        #=======================================================================
        dkey1 = 'finv_gPoly_id_dxind'
        dxind = pd.concat(finv_gkey_df_d)
        
        #check index
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
                     ):
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
        if method == 'points':
            res_d = self.sa_get(meth='get_rsamps_lib', logger=log, dkey=dkey, write=False, 
                                finv_lib=finv_aggS_lib, idfn=gcn)
        

        #=======================================================================
        # use mean depths from true assets (for error isolation only)
        #=======================================================================
        elif method == 'true_mean':
 
            #===================================================================
            # #get the true depths
            #===================================================================
            #just the true finvs
            finvT_lib = dict()
            for studyArea, lay_d in finv_aggS_lib.items():
                finvT_lib[studyArea] = {0:lay_d[0]}
            
            resT_d = self.sa_get(meth='get_rsamps_lib', logger=log, dkey=dkey, write=False, 
                            finv_lib=finvT_lib, idfn=gcn)
            
            """
            resT_d.keys()
            """
            #===================================================================
            # get means
            #===================================================================
            fgdir_dxind = self.retrieve('fgdir_dxind')
            res_d = dict()
            
            for studyArea, lay_d in finv_aggS_lib.items():
                
                res_df = None
                for grid_size, vlay in lay_d.items():
                    log.info('true_mean on %s.%s'%(studyArea, grid_size))
                    
                    rsampT_df = resT_d[studyArea].copy()
                    
                    #===========================================================
                    # trues
                    #===========================================================
                    if grid_size == 0:
                        rdf = rsampT_df
                        
                    #===========================================================
                    # gridded
                    #===========================================================
                    else:
                        #=======================================================
                        # #join gids to trues
                        #=======================================================
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
                        assert len(miss_l)==0
                        
                    #===========================================================
                    # join
                    #===========================================================
                    if res_df is None:
                        res_df = rdf
                    else:
                        res_df = res_df.append(rdf)
                #===============================================================
                # wrap study area
                #===============================================================
                res_d[studyArea] = res_df
            
            #===================================================================
            # wrap true_mean
            #===================================================================
            log.info('got true_means on %i study areas'%len(res_d))
 
 
        #=======================================================================
        # shape into dxind
        #=======================================================================
        dxind1 = pd.concat(res_d)
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
        """TODO: make a metatable for all the events then populate the names with something pretty and consistent"""
        idf = dxser.index.to_frame().reset_index(drop=True)
        idf['event'] = idf['event'].str.slice(start=5, stop=25).str.replace('_', '')
        
        dxser.index = pd.MultiIndex.from_frame(idf)
        #=======================================================================
        # #re-org
        #=======================================================================
        dxser = dxser.swaplevel().swaplevel(i=1,j=0).sort_index(axis=0, level=0, sort_remaining=True)
        
        dxser = dxser.sort_index(level=0, axis=0, sort_remaining=True)
        
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
    
    
 
        
    
    #===========================================================================
    # HELPERS--------
    #===========================================================================
    def xxxget_finvg(self, #special retrival for finvg as 
                  load_vlays=False, #whether to load the vlays found in fgm_ofp_d
                  logger=None,
                  ): #get and check the finvg data
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_finvg')
        #=======================================================================
        # load
        #=======================================================================
        d = self.retrieve('finv_gPoly')
        
        if not 'fgm_ofp_d' in d and 'fgdir_dxind' in d:
            raise Error('got bad keys on finvg')
 
        fgm_ofp_d, fgdir_dxind = d['fgm_ofp_d'], d['fgdir_dxind']
        
        #=======================================================================
        # check
        #=======================================================================
        miss_l = set(fgm_ofp_d.keys()).symmetric_difference(fgdir_dxind.index.get_level_values(0).unique())
        assert len(miss_l)==0
        
        for k,fp in fgm_ofp_d.items():
            assert os.path.exists(fp), k
            assert fp.endswith('.gpkg'), k
            
        #=======================================================================
        # load the layers
        #=======================================================================
        
        if load_vlays:
            vlay_d = dict()
            for k,fp in fgm_ofp_d.items(): 
                """not sure how this will work with changing slicing"""
                vlay_d[k] = self.vlay_load(fp, logger=log)
            fgm_ofp_d = vlay_d
            
            
        return fgm_ofp_d, fgdir_dxind
        

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
        #=======================================================================
        # load aoi
        #=======================================================================
        if not aoi is None and trim:
            self.load_aoi(aoi)
            
        #=======================================================================
        # load finv
        #=======================================================================
        finv_raw = self.vlay_load(finv_fp, dropZ=True, reproj=True)
        
        if not self.aoi_vlay is None:
            finv = self.slice_aoi(finv_raw)
            self.mstore.addMapLayer(finv_raw)
        
        else:
            finv = finv_raw
            
 
            
        #check
        assert self.idfn in [f.name() for f in finv.fields()], 'finv \'%s\' missing idfn \"%s\''%(finv.name(), self.idfn)
            
        self.mstore.addMapLayer(finv_raw)
        self.finv_vlay = finv
            
        
            
            
            
            
            
        #=======================================================================
        # attachments
        #=======================================================================
        self.wd_dir=wd_dir
 
        
    def get_finvs_gridPoly(self, #get a set of polygon grid finvs (for each grid_size)
                  finv_vlay=None,
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
        if finv_vlay is None: finv_vlay=self.finv_vlay
        if idfn is None: idfn=self.idfn
        
 
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
            
            df = vlay_get_fdf(jvlay, logger=log).drop('fid', axis=1).set_index(idfn)
            
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
        tgvlay2 = self.deletecolumn(tgvlay1, list(fnl),  logger=log)
        self.mstore.addMapLayer(tgvlay2)
        
        
        #add the grid_size
        tgvlay3 = self.fieldcalculator(tgvlay2, grid_size, fieldName='grid_size', 
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
            res_d[aggLevel] = self.get_rsamps(finv_vlay=finv_vlay, logger=self.logger.getChild('aL%i'%aggLevel),
                            **kwargs)
            
        #=======================================================================
        # combine
        #=======================================================================
        dxind = pd.concat(res_d)
        dxind.index.set_names('grid_size', level=0, inplace=True)
 
        assert dxind.notna().all().all(), 'zeros replaced at lowset level'
        
        dry_bx = dxind<=0
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i vlays and %i rasters w/ %i/%i dry'%(
            len(finv_vlay_d), len(dxind.columns), dry_bx.sum().sum(), dxind.size))
        
        try:
            self.session.check_mindex(dxind.index)
        except Exception as e:
            raise Error('%s failed w/ %s'%(self.name, e))
        
        return dxind
                       
        
    def get_rsamps(self, #raster samples on a single finv
                   wd_dir = None,
                   finv_vlay=None,
                   idfn=None,
                   logger=None,
                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_rsamps')
        if wd_dir is None: wd_dir = self.wd_dir
        if finv_vlay is None: finv_vlay=self.finv_vlay
        if idfn is None: idfn=self.idfn
        
        assert os.path.exists(wd_dir)
        assert 'Point' in QgsWkbTypes().displayString(finv_vlay.wkbType())
        #=======================================================================
        # retrieve
        #=======================================================================
        fns = [e for e in os.listdir(wd_dir) if e.endswith('.tif')]
        
        log.info('on %i rlays \n    %s'%(len(fns), fns))
        
        res_df = None
        for i, fn in enumerate(fns):
            rname = fn.replace('.tif', '')
            log.debug('%i %s'%(i, fn))
            
            vlay_samps = self.rastersampling(finv_vlay, os.path.join(wd_dir, fn), logger=log, pfx='samp_')
            self.mstore.addMapLayer(vlay_samps)
            
            #retrive and clean
            df = vlay_get_fdf(vlay_samps, logger=log).drop('fid', axis=1, errors='ignore').rename(columns={'samp_1':rname})
            
            
            
            #force type
            assert idfn in df.columns, 'missing key \'%s\' on %s'%(idfn, finv_vlay.name())
            df.loc[:, idfn] = df[idfn].astype(np.int64)
            
            assert df[idfn].is_unique, finv_vlay.name()
            
            #promote columns to multindex
            #df.index = pd.MultiIndex.from_frame(df.loc[:, [idfn, 'grid_size']])
 
            #res_d[rname] = df.drop([idfn, 'grid_size'], axis=1)
            
            rdf = df.set_index(idfn).drop('grid_size', axis=1)
            if res_df is None:
                res_df = rdf
            else:
                res_df = res_df.join(rdf, how='outer')
            
            """
            view(df)
            view(finv_vlay)
            view(dxind)
            """
        
        #=======================================================================
        # fill zeros
        #=======================================================================
        res_df1 = res_df.fillna(0.0)
        #=======================================================================
        # wrap
        #=======================================================================
 
        #res_df = pd.concat(res_d.values(), axis=0).sort_index()
        assert res_df.index.is_unique
        
        
        log.info('finished on %s and %i rasters w/ %i/%i dry'%(
            finv_vlay.name(), len(fns), res_df.isna().sum().sum(), res_df.size))
        #=======================================================================
        # try:
        #     self.session.check_mindex(dxind.index)
        # except Exception as e:
        #     raise Error('%s failed index check \n    %s'%(self.name, e))
        #=======================================================================
            
        
        return res_df1.sort_index()
    
 
        