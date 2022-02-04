'''
Created on Jan. 18, 2022

@author: cefect



'''
import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np
idx = pd.IndexSlice



from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo

 
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
            'rsamps':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.rsamps_sa(**kwargs),
                },
            
            'finvg':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.finvg_sa(**kwargs),
                },
            'rloss':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.rloss_build(**kwargs),
                },
            'tloss':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.tloss_build(**kwargs),
                },
            'errs':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.errs_build(**kwargs),
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
    # ANALYSIS-------------
    #===========================================================================
    
    def plot_depths(self,
                    plot_fign = 'studyArea',
                    plot_rown = 'grid_size', 
                    plot_coln = 'event',
 
                    
                    
                    xlims = (0,2),
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_depths')
        
        #retrieve child data
        #fgm_ofp_d, fgdir_dxind = self.get_finvg()
        serx = self.retrieve('rsamps')
        mdex = serx.index
        
        log.info('on %i'%len(serx))
        plt.close('all')
        #=======================================================================
        # loop on studyAreas
        #=======================================================================
        res_d = dict()
        for i, (sName, gsx1r) in enumerate(serx.groupby(level=plot_fign)):
            gsx1 = gsx1r.droplevel(plot_fign)
     
            
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
                #prep
                gsx2 = gsx2r.droplevel([plot_rown, plot_coln, plot_fign])
                ax = ax_d[row_key][col_key]
                
                #plot
                ar = gsx2.dropna().values
                ax.hist(ar, color='blue', alpha=0.3, label=row_key, density=True, bins=30, range=xlims)
                
                #label
                meta_d = {
                    plot_rown:row_key,
                    'wet':len(ar), 'dry':gsx2.isna().sum(), 'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
                
                txt = '\n'.join(['%s=%.2f'%(k,v) for k,v in meta_d.items()])
                ax.text(0.5, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='blue')
                
                """
                plt.show()
                """
                
                #style                    
                ax.set_xlim(xlims)
                
                #first columns
                if col_key == mdex.unique(plot_coln)[0]:
                    ax.set_ylabel('%s=%s'%(plot_rown, row_key))
                
 
                #first row
                if row_key == mdex.unique(plot_rown)[0]:
                    pass
                    #ax.set_title('event \'%s\''%(rlayName))
                    
                #last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel('depth (m)')
                        
            #===================================================================
            # wrap figure
            #===================================================================
            fig.suptitle('depths for studyArea \'%s\''%sName)
            
            res_d[sName]= self.output_fig(fig, fname='depths_%s_%s'%(sName, self.longname))
            
        #=======================================================================
        # warp
        #=======================================================================
        log.info('finished writing %i figures'%len(res_d))
        
        return res_d
                    
 
            
            
    def plot_totals_bars(self, #barchart of total losses
                    dkey = 'tloss', #dxind to plot
                    lossType = 'tl', 
                    #plot_fign = 'studyArea',
                    plot_rown = 'studyArea', 
                    plot_coln = 'vid',
                    plot_colr = 'grid_size',
                    plot_bgrp = 'event',
                    
                    #plot style
                    colorMap=None,
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
                                    set_ax_title=True,
                                    )
        fig.suptitle('%s summed'%lossType)
        
        #get colors
        cvals = dxser.index.unique(plot_colr)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        #===================================================================
        # loop and plot
        #===================================================================
        for (row_key, col_key), gser1r in dxser.groupby(level=[plot_rown, plot_coln]):
            
            #data setup
            gser1 = gser1r.droplevel([plot_rown, plot_coln])
 
            #subplot setup 
            ax = ax_d[row_key][col_key]
            
            
            #===================================================================
            # loop on colors
            #===================================================================
            for i, (ckey, gser2r) in enumerate(gser1.groupby(level=plot_colr)):
                
                #get data
                gser2 = gser2r.droplevel(plot_colr)
 
                tlsum_ser = gser2.groupby(plot_bgrp).sum()
                
                #===============================================================
                # #formatters.
                #===============================================================
                #labels
                if row_key == mdex.unique(plot_rown)[-1]: #last row
                    tick_label = ['e%i'%i for i in range(0,len(tlsum_ser))]
 
                else:
                    tick_label=None
                    
                #widths
                bar_cnt = len(mdex.unique(plot_colr))*len(tlsum_ser)
                width = 0.9/float(bar_cnt)
                
                #===============================================================
                # #add bars
                #===============================================================
                ax.bar(
                    np.linspace(0,1,num=len(tlsum_ser)) + width*i, #xlocation of bars
                    tlsum_ser.values, #heights
                    width=width,
                    align='center',
                    color=newColor_d[ckey],
                    label='%s=%s'%(plot_colr, ckey),
                    alpha=0.5,
                    tick_label=tick_label, #just adding at the end
                    )
                
            #===============================================================
            # #wrap format subplot
            #===============================================================
            
            
            #first row
            if row_key==mdex.unique(plot_rown)[0]:
                
                #last col
                if col_key == mdex.unique(plot_coln)[-1]:
                    ax.legend()
                    
            #first col
            if col_key == mdex.unique(plot_coln)[0]:
                ax.set_ylabel(dkey)
                    

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
            
            fig.suptitle('%s of %s for %s \'%s\''%(ycoln, lossType, plot_fign, fig_key))
        

        
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
            self.output_fig(fig, fname='box_%s_%s_%s_%s'%(lossType, ycoln, fig_key, self.longname))
                    

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
 
    def plot_errs_scatter(self, #scatter plot of error-like data
                    #data control   
                    dkey='errs',
                    
                    lossType='rl',
                    ycoln = 'delta',
                    xcoln = 'depth',
                       
                    #figure config
                    folder_varn = 'studyArea',
                    plot_fign = 'event',
                    plot_rown = 'grid_size', 
                    plot_coln = 'vid',
                    plot_colr = None,
                    #plot_bgrp = 'event',
                    out_dir = None,
                    
                    plot_vf=False, #plot the vf
                    
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
        if ylabel is None: ylabel = '%s of %s'%(ycoln, lossType.upper())
        
        if plot_vf:
            assert lossType=='rl'
            
        if plot_colr is None: plot_colr=plot_rown
        if out_dir is None: out_dir = os.path.join(self.out_dir, 'errs_scatter')
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
                # loop on axis row/column (and colors)
                #===================================================================
                for (row_key, col_key, ckey), gser1r in gser0r.groupby(level=[plot_rown, plot_coln, plot_colr]):
                    
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
                    
                    #join to xvalues
                    """nulls are dry"""
                    dxind = gser1r.to_frame().join(xser, on=xser.index.name)
                    
                    
                    
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
                    ax.scatter(x=dxind[xcoln].values, y=dxind[ycoln].values, 
                               color=newColor_d[ckey], s=10, marker='.', alpha=0.8,
                                   label='%s=%s'%(plot_colr, ckey))
                    
                    #===========================================================
                    # add text
                    #===========================================================

                    if add_text:
                        meta_d = {'cnt':len(dxind), 'dry_cnt':dxind['depth'].isna().sum(),
                              'mean':gser1r.mean(), 'min':gser1r.min(), 'max':gser1r.max(),
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
                                fmt='png', logger=log)
            """
            fig.show()
            """
                    

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
    
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
        fgm_vlay_d, fgdir_dxind = self.get_finvg(load_vlays=True)
        
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
            fgm_vlay = fgm_vlay_d[keys_d['studyArea']]
            
            #get key dictionary off layer
            fgm_df = vlay_get_fdf(fgm_vlay, logger=log)
            
            fgm_ser1 = fgm_df.loc[fgm_df['grid_size']==keys_d['grid_size'], ['fid', 'gid']].set_index('fid').iloc[:,0]
            
            #check key match
            miss_l = set(gdx.index.unique('gid')).difference(fgm_ser1.values)
            assert len(miss_l)==0, 'missing %i entries found in dxcol but not in fgm_vlay: \n    %s'%(len(miss_l), miss_l)
            
            #get fid match
            gf_all_d = {v:k for k,v in fgm_ser1.to_dict().items()} #all the keys
            gf_d = {gid:gf_all_d[gid] for gid in gdx.index.unique('gid')} #just those in the index
            

            #get geometries
            request = QgsFeatureRequest().setFilterFids(list(gf_d.values()))
            geo_d = vlay_get_geo(fgm_vlay, request=request)
            
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
            gdf2 = gdf1.join(pd.Series(gf_d, name='fid'), on='gid').set_index('fid')
            
            """
            view(gdf2)
            """
            #===================================================================
            # build layer
            #===================================================================
            layname = '_'.join([str(e).replace('_','') for e in keys_d.values()])
            vlay = self.vlay_new_df(gdf2, geo_d=geo_d, layname=layname, logger=log, 
                                    crs=fgm_vlay.crs(), #session crs does not match studyAreas
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

    def errs_build(self, #get the errors (gridded - true)
                    dkey=None,
                     prec=None,
                    ):
        """
        delta: grid - true
        errRel: delta/true
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('errs_build')
        assert dkey=='errs'
        if prec is None: prec=self.prec
        gcn = self.gcn
 
        #=======================================================================
        # retriever
        #=======================================================================
        tl_dx = self.retrieve('tloss')
        
        tlnames_d= {lvlName:i for i, lvlName in enumerate(tl_dx.index.names)}
        
        #identify totalLoss columns (and their vid)
        #tlcoln_d = {c:int(c.replace('tloss','')) for c in tl_dxind.columns if c.endswith('tloss')}

        _, fgdir_dxind = self.get_finvg()

        #=======================================================================
        # clean data
        #=======================================================================
        #tl_dxind1 = tl_dxind.loc[:, tlcoln_d.keys()]
        #tl_dxind1 = tl_dxind1.rename(columns = {c:tlcoln_d[c] for c in tl_dxind1.columns})
        """
        view(tl_dxind)
        view(tl_dxind1)
        """
        vid_l = tl_dx.drop('meta', axis=1, level=0).columns.unique(1).to_list()
        #=======================================================================
        # #loop on each grid_size, studyArea, event
        #=======================================================================
        res_d, meta_d = dict(), dict()
        lvls = [0,1,2]
        
        colx =  pd.MultiIndex.from_product([['delta', 'errRel', 'grid', 'true'], vid_l],
                             names=['compare', 'vid']).sortlevel(0, sort_remaining=True)[0]
                             
        
        for lossType in tl_dx.drop('meta', axis=1, level=0).columns.unique(0).to_list():
            
            res_dx = None
            for keys, gdxind in tl_dx.groupby(level=lvls, axis=0):
                keys_d = dict(zip([tl_dx.index.names[i] for i in lvls], keys))
                
                #===================================================================
                # trues
                #===================================================================
                if keys_d['grid_size']==0:
                    
                    #err_dx = pd.concat([gdxind], keys=['true'], names=['tlType'], axis=1)
     
                    #start with all zeros
                    err_dx = pd.DataFrame(0, index=gdxind.index,columns=colx)
                    
                    #set trues
                    err_dx.loc[:, idx['true', :]] = gdxind.loc[:, idx[lossType, :]].values
                    
                    #set grids
                    err_dx.loc[:, idx['grid', :]] = np.nan
 
                    
                #===================================================================
                # gridded
                #===================================================================
                else:
                    

                    
                    #===================================================================
                    # #get trues
                    #===================================================================
                    true_dxind0 = tl_dx.loc[idx[0, keys[1], keys[2], :],idx[lossType,:]].droplevel(0, axis=1)
     
                    #relabel to true ids
                    true_dxind0.index.set_names('id', level=tlnames_d['gid'], inplace=True)
                    
                    #get true ids (many ids to 1 gid))
                    id_gid_ser = fgdir_dxind.loc[idx[keys_d['studyArea'], :], keys_d['grid_size']].rename(gcn).droplevel(0)
                    
                    
                    if not id_gid_ser.index.is_unique:
                        #id_gid_ser.to_frame().loc[id_gid_ser.index.duplicated(keep=False), :]
                        raise Error('bad index on %s'%keys_d)
                    
                    #join gids
                    true_dxind1 = true_dxind0.join(id_gid_ser).set_index(gcn, append=True).droplevel(lvls).sort_index()
                    
                    #get totals per gid
                    if lossType == 'tl': #true values are sum of each child
                        true_df0 = true_dxind1.groupby(gcn).sum()
                    elif lossType=='rl': #true values are the average of family
                        true_df0 = true_dxind1.groupby(gcn).mean()
                    else:
                        raise Error('bad lossType')
    
                    assert true_df0.index.is_unique
                    
                    #===========================================================
                    # get gridded
                    #===========================================================
                    #clean the gridded values (gid:vid:totalLoss)
                    gdf = gdxind.droplevel(lvls).loc[:, idx[lossType,:]].droplevel(0, axis=1).sort_index()
                    assert gdf.index.is_unique, keys_d
                    
                    #replace any empties in gridded
                    miss_l = set(true_df0.index).difference(gdf.index) #those gids with trues but no gridded
 
 
 
                    
                    
                    #reshape like the gridded and add zeros
                    """
                    those missing in either were dry (missees)
                    
                    set_info(gdf.index, true_df0.index)
                    """
                    #true_df1 = true_df0.reindex(gdf.index).fillna(0)
                    
                    #outer join (
                    dxcol1 = pd.concat([gdf, true_df0], keys=['grid', 'true'], names=colx.names, axis=1).fillna(0)
 
                    #===================================================================
                    # deltas (gridded - true)
                    #===================================================================
 
                    dlta_dx = dxcol1.loc[:, idx['grid', :]].subtract(dxcol1.loc[:, idx['true', :]].values)
                    dlta_dx.columns.set_levels(['delta'], level=0, inplace=True)
      
                    dxcol2 = dxcol1.join(dlta_dx)
                    """
                    view(dxcol2)
                    """
                                                        
                    #===============================================================
                    # relative errors (delta/true)
                    #===============================================================
                    rel_dx = dxcol2.loc[:, idx['delta', :]].divide(dxcol2.loc[:, idx['true', :]].values).fillna(0)
                    rel_dx.columns.set_levels(['errRel'], level=0, inplace=True)
                    
                    # join back index
                    dxcol3 = dxcol2.join(rel_dx).sort_index()
                    
 
                    #===============================================================
                    #clean
                    #===============================================================
                    #revert to mindex
                    err_dx = dxcol3.copy()
                    """new index has more gids as we've done an outer join w/ the trues
                    err_dx.index = gdxind.index"""
                    
                    
                    #create a new mindex for the expanded data
                    err_dx.index = pd.MultiIndex.from_product(
                                [[keys[0]], [keys[1]],[keys[2]], err_dx.index],
                                names=gdxind.index.names,
                                )
                    
                    err_dx = err_dx.sort_index(level=3)
                    err_dx = err_dx.sort_index(level=0, sort_remaining=True, axis=1)
                    
                    assert err_dx.notna().all().all()
                    
 
                #===================================================================
                # wrap groupby loop
                #===================================================================
                log.debug("got %s"%str(err_dx.shape))
                
 
                if not np.array_equal(err_dx.columns, colx):
                    raise Error('column mismatch on %s'%keys_d)
                
                if res_dx is None:
                    res_dx=err_dx
                else:
                    res_dx = res_dx.append(err_dx, verify_integrity=True)
            
            #===================================================================
            # wrap lossType loop
            #===================================================================
            res_d[lossType] = res_dx
            
            meta_d[lossType] = pd.concat({
                        'max':res_dx.groupby(level='grid_size').max(),
                        'count':res_dx.groupby(level='grid_size').count(),
                        'sum':res_dx.groupby(level='grid_size').sum(),
                        }, axis=1)
                
        #=======================================================================
        # wrap
        #=======================================================================
        res_dx = pd.concat(res_d, axis=1)
        res_dx.columns = res_dx.columns.remove_unused_levels().set_names('lossType', level=0)
        
        #reporting
        
        mdf = pd.concat(meta_d)
        
        
        log.info('finished w/ %s and totalErrors: \n%s'%(
            str(res_dx.shape), mdf))
        
        

                
        #=======================================================================
        # write
        #=======================================================================
 
        
        self.ofp_d[dkey] = self.write_pick(res_dx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                                   logger=log)

        return res_dx
 
 
    
    
    def tloss_build(self, #get the total loss
                    dkey=None,
                    prec=2,
                    ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('tloss_build')
        assert dkey=='tloss'
        if prec is None: prec=self.prec
        gcn = self.gcn
        scale_cn = 'id_cnt'
        #=======================================================================
        # retriever
        #=======================================================================
        rl_dxind = self.retrieve('rloss')
        
        rlnames_d= {lvlName:i for i, lvlName in enumerate(rl_dxind.index.names)}
        
        _, fgdir_dxind = self.get_finvg()
        
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
            'meta':dxind1.loc[:, ['depth', scale_cn]],
            'rl':dxind1.loc[:, rl_dxind.drop('depth', axis=1).columns]},
            axis=1)
        
        
        
        #=======================================================================
        # rl_cols = rl_dxind.drop('depth',axis=1).columns.tolist()
        # 
        # new_rl_cols= ['%i_rl'%e for e in rl_cols]
        # dxind2 = dxind1.rename(columns=dict(zip(rl_cols, new_rl_cols)))
        #=======================================================================
        
        #=======================================================================
        # calc total loss
        #=======================================================================
 
            
        tl_dx = dx1.loc[:, idx['rl', :]].multiply(
            dx1.loc[:, idx['meta', scale_cn]], axis=0
            ).round(prec)
            
        #rename the only level0 value        
        tl_dx.columns = tl_dx.columns.remove_unused_levels().set_levels(levels=['tl'], level=0)
        
        #join these in 
        dx2 = dx1.join(tl_dx)
 
        
        """
        view(dx2)
        view(rdx)
        """
        
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

    
    def rloss_build(self,
                    dkey=None,
                    prec=None, #precision for RL
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('rloss_build')
        assert dkey=='rloss'
        if prec is None: prec=self.prec
        
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        #depths
        #fgm_ofp_d, fgdir_dxind = self.get_finvg()
        dxser = self.retrieve('rsamps')
        
        #collapse data
        """should probably make this the rsamp format by default"""

        
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
 
                    
    
    def finvg_sa(self, #build the finv groups on each studyArea
                 dkey=None,
                 
                 **kwargs):
        """here we store the following in one pickle 'finvg'
            fgm_fps_d: filepaths to merged grouped finvs {name:fp}
            fgdir_dxind: directory of finv keys from raw to grouped for each studyArea
            """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('finvg_sa')
        assert dkey=='finvg'
        
        #=======================================================================
        # #run the method on all the studyAreas
        #=======================================================================
        res_d = self.sa_get(meth='get_finvsg', write=False, dkey=dkey, **kwargs)
        
        #unzip results
        finv_gkey_df_d, fgm_vlay_d = dict(), dict() 
        for k,v in res_d.items():
            finv_gkey_df_d[k], fgm_vlay_d[k]  = v
            
        
        
        #=======================================================================
        # write finv points vector layers
        #=======================================================================
 
        #setup directory
        od = os.path.join(self.wrk_dir, 'fgm_vlays')
        if not os.path.exists(od): os.makedirs(od)
        
        log.info('writing %i layers to %s'%(len(fgm_vlay_d), od))
        
        ofp_d = dict()
        for name, fgm_vlay in fgm_vlay_d.items():
            ofp_d[name] = self.vlay_write(fgm_vlay,
                                     os.path.join(od, 'fgm_%s_%s.gpkg'%(self.longname, name)),
                                     logger=log)
            
        log.debug('wrote %i'%len(ofp_d))
        
 
        
        
        #=======================================================================
        # write the dxcol and the fp_d
        #=======================================================================
        dxind = pd.concat(finv_gkey_df_d)
        
        #check index
        dxind.index.set_names('studyArea', level=0, inplace=True)
        self.check_mindex(dxind.index)

 
        
        log.info('writing \'fgdir_dxind\' (%s) and \'fgm_ofp_d\' (%i) as \'%s\''%(
            str(dxind.shape), len(ofp_d), dkey))
        
        
        #save the directory file
        finvg = {'fgm_ofp_d':ofp_d,'fgdir_dxind':dxind}
        
        self.ofp_d[dkey] = self.write_pick(finvg, 
                           os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),logger=log)
        
        """
        finvg.keys()
        """
        return finvg
        
 
    def rsamps_sa(self, #get raster samples for all finvs
                     dkey=None,
                     proj_lib=None,
                     ):
        """
        keeping these as a dict because each studyArea/event is unique
        """
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey=='rsamps', dkey
        log = self.logger.getChild('rsamps_sa')
        if proj_lib is None: proj_lib=self.proj_lib
        gcn=self.gcn
        #=======================================================================
        # child data
        #=======================================================================
        fgm_ofp_d, fgdir_dxind = self.get_finvg()
        
        log.debug('on %i'%len(fgm_ofp_d))
        
        #=======================================================================
        # update the finv
        #=======================================================================
        d = dict()
        for name, pars_d in proj_lib.items():
            d[name] = copy.deepcopy(pars_d)
            d[name]['finv_fp'] = fgm_ofp_d[name]
            d[name]['idfn'] = gcn #these finvs are keyed by gid
        
 
        res_d = self.sa_get(proj_lib=d, meth='get_rsamps', logger=log, dkey=dkey, write=False)
        
        #=======================================================================
        # shape into dxind
        #=======================================================================
        dxind1 = pd.concat(res_d, names=['studyArea', 'grid_size', gcn])
        
 
        
        dxser = dxind1.stack().rename('depth') #promote depths to index
        dxser.index.set_names('event', level=3, inplace=True) 
 
        
        dxser = dxser.swaplevel().swaplevel(i=1,j=0).sort_index(axis=0, level=0, sort_remaining=True)
        
        self.check_mindex(dxser.index)
 
        
        
        log.debug('on %i'%len(res_d))
        
        #=======================================================================
        # write
        #=======================================================================
        self.ofp_d[dkey] = self.write_pick(dxser, os.path.join(self.wrk_dir, '%s_%s.pickle'%(dkey, self.longname)),
                        logger=log)
        
        return dxser
    
    #===========================================================================
    # HELPERS--------
    #===========================================================================
    def get_finvg(self,
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
        d = self.retrieve('finvg')
        
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
        
    
        
    #===========================================================================
    # GENERICS---------------
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
        
        
        assert dkey in ['rsamps', 'finvg'], 'bad dkey %s'%dkey
        
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
 
        
    def get_finvsg(self, #get the set of finvs per aggLevel
                  finv_vlay=None,
                  grid_sizes = [5, 20, 100], #resolution in meters
                  idfn=None,
                  write_grids=True, #whether to also write the grids to file
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
        log = self.logger.getChild('get_finvsg')
        gcn = self.gcn
        if overwrite is None: overwrite=self.overwrite
        if finv_vlay is None: finv_vlay=self.finv_vlay
        if idfn is None: idfn=self.idfn
        
        #=======================================================================
        # loop and aggregate
        #=======================================================================
        fpts_d = dict() #container for resulting finv points layers
        groups_d = dict()
        meta_d = dict()
 
        log.info('on \'%s\' w/ %i: %s'%(finv_vlay.name(), len(grid_sizes), grid_sizes))
        
        for i, grid_size in enumerate(grid_sizes):
            log.info('%i/%i grid w/ %.1f'%(i+1, len(grid_sizes), grid_size))
            
            #===================================================================
            # #build the grid
            #===================================================================
            
            gvlay1 = self.creategrid(finv_vlay, spacing=grid_size, logger=log)
            self.mstore.addMapLayer(gvlay1)
            log.info('    built grid w/ %i'%gvlay1.dataProvider().featureCount())
            
            #clear all the fields
            #gvlay1a = self.deletecolumn(gvlay1, [f.name() for f in gvlay1.fields()], logger=log)
            
            #index the grid
            #gvlay2 = self.addautoincrement(gvlay1a, fieldName='gid', logger=log)
            
            gvlay2 = self.renameField(gvlay1, 'id', gcn, logger=log)
            self.mstore.addMapLayer(gvlay2)
            log.info('    renamed field \'id\':\'%s\''%gcn)
            
            #handel writing
            if write_grids:
                od = os.path.join(self.out_dir, 'grids', self.name)
                if not os.path.exists(od):os.makedirs(od)
                output = os.path.join(od, 'fgrid_%i_%s.gpkg'%(grid_size, self.longname))
                if os.path.exists(output):
                    assert overwrite
                    os.remove(output)
            else:
                output='TEMPORARY_OUTPUT'
            
            #select those w/ some assets
            gvlay2.removeSelection()
            self.createspatialindex(gvlay2)
            log.info('    selecting from grid based on intersect w/ \'%s\''%(finv_vlay.name()))
            self.selectbylocation(gvlay2, finv_vlay, logger=log)
            
 
            gvlay2b = self.saveselectedfeatures(gvlay2, logger=log, output=output)
            if not write_grids: 
                self.mstore.addMapLayer(gvlay2b)
            else:
                log.info('    wrote grid layer to %s'%output)
 
            
            #drop these to centroids
            gvlay3 = self.centroids(gvlay2b, logger=log)
            self.mstore.addMapLayer(gvlay3)
            

 
            
            #add groupsize field            
            gvlay_pts = self.fieldcalculator(gvlay3, grid_size, fieldName='grid_size', 
                                           fieldType='Integer', logger=log)
            
 
            
            log.info('    got %i pts from grid'%gvlay3.dataProvider().featureCount())
            #===================================================================
            # #copy/join over the keys
            #===================================================================
            jd = self.joinattributesbylocation(finv_vlay, gvlay2b, jvlay_fnl=gcn, 
                                               method=1, 
                                               #predicate='touches',
                                               logger=log,
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
            gpts_ser = vlay_get_fdf(gvlay_pts)[gcn]
            
            set_d = set_info(gpts_ser.values, df[gcn].values)

            
            if not len(set_d['symmetric_difference'])==0:
                del set_d['union']
                del set_d['intersection']
                log.warning('%s.%i got %i mismatched values... deleteing these grid cells\n   %s'%(
                    self.name, grid_size, len(set_d['symmetric_difference']), set_d))
                
                assert len(set_d['diff_right'])==0
                
                #get matching ids
                fid_l = gpts_ser.index[gpts_ser.isin(set_d['diff_left'])].tolist()
                gvlay_pts.removeSelection()
                gvlay_pts.selectByIds(fid_l)
                assert gvlay_pts.selectedFeatureCount()==len(set_d['diff_left'])
                
                #delete these
                gvlay_pts.invertSelection()
                gvlay_pts = self.saveselectedfeatures(gvlay_pts, logger=log)
            
            
            
            
            #===================================================================
            # wrap
            #===================================================================
            fpts_d[grid_size]=gvlay_pts
            groups_d[grid_size] = df.copy()
 
            #===================================================================
            # #meta
            #===================================================================
            mcnt_ser = df[gcn].groupby(df[gcn]).count()
            meta_d[grid_size] = {
                'total_cells':gvlay1.dataProvider().featureCount(), 
                'active_cells':gvlay2.selectedFeatureCount(),
                'max_member_cnt':mcnt_ser.max()
                }
            
            
            
            log.info('    joined w/ %s'%meta_d[grid_size])
            
        #=======================================================================
        # combine results----
        #=======================================================================
        log.info('finished on %i'%len(groups_d))
        """
        view(finvR_vlay2)
        """
        #assemble the grid id per raw asset
        finv_gkey_df = pd.concat(groups_d, axis=1).droplevel(axis=1, level=1)
        
        #=======================================================================
        # #merge points vector layers
        #=======================================================================
        #prep the raw
        finvR_vlay1 = self.fieldcalculator(finv_vlay, 0, fieldName='grid_size', fieldType='Integer', logger=log)
        finvR_vlay2 = self.fieldcalculator(finvR_vlay1,'\"{}\"'.format(idfn), fieldName=gcn, fieldType='Integer', logger=log)
        
        #finvR_vlay2 = self.renameField(finvR_vlay1, idfn, gcn, logger=log) #promote old keys to new keys
        
 
        
        #add the raw finv (to the head)
        fpts_d = {**{0:finvR_vlay2},**fpts_d}
        
        #merge
        fgm_vlay1 = self.mergevectorlayers(list(fpts_d.values()), logger=log)
        self.mstore.addMapLayer(fgm_vlay1)
        

        
        #drop some fields
        fnl = [f.name() for f in fgm_vlay1.fields()]
        fgm_vlay2 = self.deletecolumn(fgm_vlay1, list(set(fnl).difference([gcn, 'grid_size'])), logger=log)
        
        
        #force types
        fgm_vlay3 = self.vlay_field_astype(fgm_vlay2, gcn, fieldType='Integer')
        
        #=======================================================================
        # check
        #=======================================================================
        for coln, typeName in vlay_dtypes(fgm_vlay3).items():
            assert typeName=='integer', 'got bad type on \'%s\':%s'%(coln, typeName)
            
        fgm_df = vlay_get_fdf(fgm_vlay3)
        
        for coln, dtype in fgm_df.dtypes.to_dict().items():
            assert dtype==np.dtype('int64'), 'bad type on %s: %s'%(coln, dtype)
        
        
        for grid_size, fgm_gdf in fgm_df.groupby('grid_size', axis=0):
            if grid_size==0: continue
            miss_l = set(fgm_gdf[gcn].values).difference(finv_gkey_df[grid_size].values)  #those in left no tin right
            assert len(miss_l)==0, 'missing %i/%i keys on %s grid_size=%i'%(
                len(miss_l), len(fgm_gdf), self.name, grid_size)
        """
        fgm_gdf.sort_values('gid').dtypes
        view(fgm_gdf.sort_values('gid'))
        view(fgm_vlay3)
        """
 
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('merged %i gridded inventories into %i pts'%(
            len(fpts_d), fgm_vlay3.dataProvider().featureCount()))
        
        return finv_gkey_df, fgm_vlay3
        
 
            
                
        
    def get_rsamps(self,
                   wd_dir = None,
                   finv_vlay=None,
                   idfn=None,

                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_rsamps')
        if wd_dir is None: wd_dir = self.wd_dir
        if finv_vlay is None: finv_vlay=self.finv_vlay
        if idfn is None: idfn=self.idfn
        
        assert os.path.exists(wd_dir)
        
        #=======================================================================
        # retrieve
        #=======================================================================
        fns = [e for e in os.listdir(wd_dir) if e.endswith('.tif')]
        
        log.info('on %i rlays \n    %s'%(len(fns), fns))
        
        res_d = dict()
        for i, fn in enumerate(fns):
            rname = fn.replace('.tif', '')
            log.debug('%i %s'%(i, fn))
            vlay_samps = self.rastersampling(finv_vlay, os.path.join(wd_dir, fn), logger=log, pfx='samp_')
            self.mstore.addMapLayer(vlay_samps)
            
            #retrive and clean
            df = vlay_get_fdf(vlay_samps, logger=log).drop('fid', axis=1).rename(columns={'samp_1':rname})
            
            #force type
            df.loc[:, idfn] = df[idfn].astype(np.int64)
            
            #promote columns to multindex
            df.index = pd.MultiIndex.from_frame(df.loc[:, [idfn, 'grid_size']])
 
            res_d[rname] = df.drop([idfn, 'grid_size'], axis=1)
            
            """
            view(finv_vlay)
            view(vlay_samps)
            """
            
        #=======================================================================
        # wrap
        #=======================================================================
        dxind = pd.concat(res_d, axis=1, keys=None).droplevel(0, axis=1).swaplevel(axis=0)
        
        try:
            self.session.check_mindex(dxind.index)
        except Exception as e:
            raise Error('%s failed index check \n    %s'%(self.name, e))
            
        
        return dxind
    
 
        