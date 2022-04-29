'''
Created on Apr. 26, 2022

@author: cefect

visualizetion on raster calcs
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
start = datetime.datetime.now()
print('start at %s' % start)

import qgis.core
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal
idx = pd.IndexSlice
#===============================================================================
# setup matplotlib
#===============================================================================
 
import matplotlib
matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt

#set teh styles
plt.style.use('default')

#font
matplotlib_font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 10 #set the figure title size
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)

from agg.hyd.rast.hr_scripts import RastRun, view, Catalog, Error
from hp.plot import Plotr
from hp.gdal import rlay_to_array, getRasterMetadata
from hp.basic import set_info, get_dict_str
#from hp.animation import capture_images

class RasterAnalysis(RastRun, Plotr): #analysis of model results

    
        #colormap per data type
    colorMap_d = {
 
        'studyArea':'Dark2',
 
        'resolution':'copper',
        'dkey':'Pastel2',
        'dsampStage':'Set1'
 
        }
    
    def __init__(self,
 
                 name='rast_analy',
                 plt=None,
                 exit_summary=False,
                 **kwargs):
        
        data_retrieve_hndls = {
            'catalog':{
                #probably best not to have a compliled version of this
                'build':lambda **kwargs:self.build_catalog(**kwargs), #
                },
            
            }
        
        self.plt=plt
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls,name=name,init_plt_d=None,
                         exit_summary=exit_summary,**kwargs)
        
    def runRastAnalysis(self,
                        ):
        self.retrieve('catalog')
        
        
    def build_catalog(self,
                      dkey='catalog',
                      catalog_fp=None,
                      logger=None,
                      **kwargs):
        if logger is None: logger=self.logger
        assert dkey=='catalog'
        if catalog_fp is None: catalog_fp=self.catalog_fp
        
        return Catalog(catalog_fp=catalog_fp, logger=logger, overwrite=False, **kwargs).get()
 
    def plot_rvalues(self, #flexible plotting of raster values
                  
                    #data control
                    drop_zeros=True, 
 
                    #data 
                    fp_serx=None, #catalog
                    debug_max_len=1e4,
                                      
                    
                    #plot config
                    plot_type='gaussian_kde', 
                    plot_rown='studyArea',
                    plot_coln='dsampStage',
                    plot_colr='resolution',                    
                    plot_bgrp=None, #grouping (for plotType==bars)

                     
                    #histwargs
                    bins=20, rwidth=0.9, 
                    hrange=None, #slicing the data (similar to xlims.. but this affects the calcs)
                    mean_line=False, #plot a vertical line on the mean
                    density=True,
 
 
                    #meta labelling
                    meta_txt=True, #add meta info to plot as text
                    meta_func = lambda meta_d={}, **kwargs:meta_d, #lambda for calculating additional meta information (add_meta=True)        
                    write_meta=False, #write all the meta info to a csv            
                    
                    write_stats=True, #calc and write stats from raster arrays
                    
                    #plot style                    
                    colorMap=None, title=None, val_lab='depth (m)', 
                    grid=True,
                    sharey='row',sharex='row',
                    xlims=None, #also see hrange
                    ylims=None,
                    figsize=None,
                    yscale='linear',
                    
                    #output
                    fmt='svg',
 
                    **kwargs):
        """"
        similar to plot_dkey_mat (more flexible)
        similar to plot_err_mat (1 dkey only)
        
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_rvalues')
 
        resCn, saCn = self.resCn, self.saCn
 
        #retrieve data
        """just the catalog... we load each raster in the loop"""
        if fp_serx is None:
            fp_serx = self.retrieve('catalog')['rlay_fp']
 
            
        #plot keys
        if plot_colr is None: 
            plot_colr=plot_bgrp
        
        if plot_colr is None: 
            plot_colr=plot_rown
            
        if plot_bgrp is None:
            plot_bgrp = plot_colr
            
        #=======================================================================
        # key checks
        #=======================================================================
 
        assert not plot_rown==plot_coln
            
        #plot style
                 
        if title is None:
            title = 'Depth Raster Values'
                
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        
 
        
        if plot_type in ['violin', 'bar']:
            assert xlims is None
            
        #=======================================================================
        # data prep
        #=======================================================================
        if plot_coln is None:
            """add a dummy level for c onsistent indexing"""
            plot_coln = ''
            fp_serx = pd.concat({plot_coln:fp_serx}, axis=0, names=[plot_coln])
        
 
        #=======================================================================
        # setup the figure
        #=======================================================================
        mdex = fp_serx.index
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, 
                                    sharex=sharex,  
                                    fig_id=0,
                                    set_ax_title=True,figsize=figsize,
                                    )
     
 

        assert isinstance(fig, matplotlib.figure.Figure)
        fig.suptitle(title)
        #=======================================================================
        # #get colors
        #=======================================================================
 
        ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        stats_dx = None
        meta_dx=None
        for gkeys, gdx0 in fp_serx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
            assert len(gdx0.index.unique('resolution'))==len(gdx0.index.get_level_values('resolution'))
            
            meta_d = {'layers':len(gdx0), 'drop_zeros':drop_zeros}
            #===================================================================
            # build data-------
            #===================================================================
            #===================================================================
            # get filepaths
            #===================================================================
            gdf0 = gdx0.droplevel(list(range(1,5)))
            
            #set order
            if plot_type=='gaussian_kde':
                fp_d = gdf0.sort_index(ascending=False).to_dict()
            else:
                fp_d = gdf0.to_dict()
                
            
            
            #===================================================================
            # loop and ubild
            #===================================================================
            log.info('for %s looping on %i rasters'%(keys_d, len(fp_d)))
            stats_d=dict()
            data_d = dict()
            zcnt=0
            drop_cnt=0
            for resolution, fp in fp_d.items():
                keys_d[resCn] = resolution
                #===============================================================
                # get the values
                #===============================================================
                ar_raw = rlay_to_array(fp)
                
                ser1 = pd.Series(ar_raw.reshape((1,-1))[0]).dropna()
                
                if write_stats:
                    stats_d[resolution] = {'%s_raw'%k:getattr(ser1, k)() for k in ['mean', 'max', 'count', 'min']}
                    
                #remove zeros
                bx = ser1==0.0
                if drop_zeros:                    
                    ser1 = ser1.loc[~bx]
                    stats_d[resolution].update({'zero_count':bx.sum()})

                if write_stats:
                    stats_d[resolution].update({'%s_noZeros'%k:getattr(ser1, k)() for k in ['mean', 'max', 'count', 'min']})
                    
                #counts
                zcnt+= bx.sum()
                if not len(ser1)>100:
                    if plot_type in ['gaussian_kde']:
                        log.warning('%s got %i entries... skipping'%(keys_d, len(ser1)))
                        drop_cnt+=1
                        continue
                
                #reduce
                if not debug_max_len is None:
                    if len(ser1)>debug_max_len:
                        log.warning('reducing from %i to %i'%(len(ser1), debug_max_len))
                        meta_d['raw_cnt'] = len(ser1)
                        ser1 = ser1.sample(int(debug_max_len)) #get a random sample of these
                        
                data_d[resolution] = ser1.values
            
            #===================================================================
            # post
            #===================================================================
            del keys_d[resCn]
            meta_d.update ({'zero_count':zcnt, 'lay_drop_cnt':drop_cnt})
            

            #===================================================================
            # stats
            #===================================================================
            if write_stats:
                sdfi = pd.DataFrame.from_dict(stats_d).T
                sdfi.index.name=resCn
                for k,v in keys_d.items():
                    sdfi[k]=v
                
                sdxi = sdfi.set_index(list(keys_d.keys()), append=True)
                    
                if stats_dx is None:
                    stats_dx = sdxi
                else:
                    stats_dx = stats_dx.append(sdxi)
            
            #===============================================================
            # plot------
            #===============================================================
            cdi = {k:color_d[k] for k in data_d.keys()} #subset and reorder
 
            """setup to plot a set... but just using the loop here for memory efficiency"""
            md1 = self.ax_data(ax, data_d,
                           plot_type=plot_type, 
                           bins=bins, rwidth=rwidth, 
                           mean_line=None, hrange=hrange, density=density,
                           color_d=cdi, 
                           logger=log,
                           violin_line_kwargs=dict(color='red', alpha=0.75, linewidth=0.75),
                           label_key=plot_bgrp, **kwargs) 
            """
                fig.show()
                """
 
             
            #===================================================================
            # add plots--------
            #===================================================================
            if mean_line:
                raise Error('not implemented')
                mval =gdx0.mean().mean()
            else: mval=None 
            

 
            meta_d.update(md1)
            
            """this is usually too long
            labels = ['%s=%s'%(plot_bgrp, k) for k in data_d.keys()]"""
            labels = [str(v) for v in data_d.keys()]
 
            #===================================================================
            # post format 
            #===================================================================
            
            ax.set_title(' & '.join(['%s:%s' % (k, v) for (k, v) in keys_d.items()]))
            #===================================================================
            # meta  text
            #===================================================================
            """for bars, this ignores the bgrp key"""
            meta_d = meta_func(logger=log, meta_d=meta_d, pred_ser=gdx0)
            if meta_txt:
                ax.text(0.1, 0.9, get_dict_str(meta_d), 
                        bbox=dict(boxstyle="round,pad=0.05", fc="white", lw=0.0,alpha=0.9 ),
                        transform=ax.transAxes, va='top', fontsize=8, color='black')
            
            
            #===================================================================
            # collect meta 
            #===================================================================
            meta_serx = pd.Series(meta_d, name=gkeys)
            
            if meta_dx is None:
                meta_dx = meta_serx.to_frame().T
                meta_dx.index.set_names(keys_d.keys(), inplace=True)
            else:
                meta_dx = meta_dx.append(meta_serx)
                
 
                
        #===============================================================
        # post format subplot ----------
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                if grid: ax.grid()
                

                
                if plot_type in ['box', 'violin']:
                    ax.set_xticklabels([])
                    
                
                
                if not xlims is None:
                    ax.set_xlim(xlims)
                    
                if not ylims is None:
                    ax.set_ylim(ylims)
                    
                #chang eto log scale
                ax.set_yscale(yscale)
                
                
                # first row
                if row_key == row_keys[0]:
                    #last col
                    if col_key == col_keys[-1]:
                        if plot_type in ['hist', 'gaussian_kde', 
                                         #'violin' #no handles 
                                         ]:
                            ax.legend()
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        if density:
                            ax.set_ylabel('density')
                        else:
                            ax.set_ylabel('count')
                    elif plot_type in ['box', 'violin']:
                        if not val_lab is None: ax.set_ylabel(val_lab)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        if not val_lab is None: ax.set_xlabel(val_lab)
                    elif plot_type in ['violin', 'box']:
                        ax.set_xticks(np.arange(1, len(labels) + 1))
                        ax.set_xticklabels(labels)
                    #last col
                    if col_key == col_keys[-1]:
                        pass
                        
                    
 
 
        #=======================================================================
        # wrap---------
        #=======================================================================
        log.info('finsihed')
        """
        fig.show()
        plt.show()
        """
 
        fname = 'rvalues_%s_%s_%sX%s_%s_%s' % (
            title,plot_type, plot_rown, plot_coln, val_lab, self.longname)
                
        fname = fname.replace('=', '-').replace(' ','').replace('\'','')
        
        try:
            if write_meta:
                
                ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
                meta_dx.to_csv(ofp)
                log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
                
            if write_stats:
                ofp =os.path.join(self.out_dir, fname+'_stats.csv')
                stats_dx.to_csv(ofp)
                log.info('wrote stats_dx %s to \n    %s'%(str(stats_dx.shape), ofp))
        except Exception as e:
            log.error('failed to write meta or stats data w/ \n    %s'%e)
               
        
        return self.output_fig(fig, fname=fname, fmt=fmt)
 
 

    
    def plot_vsResolution(self, #progression against resolution plots
                         #data
                         dx_raw=None, #combined model results
                         coln = 'MEAN', #variable to plot against resolution
                         
                         #plot control
                        plot_type='line', 
                        plot_rown='studyArea',
                        plot_coln=None,
                        plot_colr=None,
                        plot_bgrp='dsampStage', #sub-group onto an axis
                        
                        
                           colorMap=None,
                         plot_kwargs = dict(marker='x'),
                         title=None,xlabel=None,ylabel=None,
                         xscale='log',
                         xlims=None,
                         
                         
                         #plot control [matrix]

                         sharey='none',sharex='col',
                         
                         
                         logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is  None: logger=self.logger
        log = logger.getChild('plot_progression')
        resCn, saCn = self.resCn, self.saCn
        
        if plot_colr is None: plot_colr=plot_bgrp
        if colorMap is None: colorMap=self.colorMap_d[plot_colr]
        
        if xlabel is None: xlabel = resCn
        if ylabel is None: ylabel=coln
        #=======================================================================
        # retrival
        #=======================================================================
        if dx_raw is None: dx_raw = self.retrieve('catalog')
        """
        view(dx_raw)
        dx_raw.index.names
        """
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert coln in dx_raw.columns, coln
 
        
        #=======================================================================
        # data prep----
        #=======================================================================
        log.info('on %i'%len(dx_raw))
        serx = dx_raw[coln]
        
        if plot_coln is None:
            """add a dummy level for c onsistent indexing"""
            plot_coln = ''
            serx = pd.concat({plot_coln:serx}, axis=0, names=[plot_coln])
 
        mdex = serx.index
        gcols = set()
        for c in [plot_bgrp, plot_coln, plot_rown]:
            if not c is None: 
                gcols.add(c)
                assert c in mdex.names, c
        gcols = list(gcols)
        #=======================================================================
        # plot defaults
        #=======================================================================
        #title
        if title is None:
            title='%s vs. %s'%(coln, resCn)
        #get colors
        ckeys = mdex.unique(plot_colr)
        
        """nasty workaround to get colors to match w/ hyd""" 
        if plot_colr =='dsampStage':
            ckeys = ['none'] + ckeys.values.tolist()
        
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        if plot_coln is None:
            col_keys = None
        else:
            col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey,sharex=sharex,  
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
 
        fig.suptitle(title)
            
            
            
        """
        fig.show()
        """

            
        #=======================================================================
        # loop and plot------
        #=======================================================================
        
        for gkeys, gsx0 in serx.groupby(level=gcols):
            keys_d = dict(zip(gcols, gkeys))
            log.info('on %s w/ %i'%(keys_d, len(gsx0)))
            ax = ax_d[keys_d[plot_rown]][keys_d[plot_coln]]
            #===================================================================
            # data prep
            #===================================================================
            xar = gsx0.index.get_level_values(resCn).values #resolutions
            yar = gsx0.values
            color=color_d[keys_d[plot_colr]]
            #===================================================================
            # plot
            #===================================================================
            if plot_type=='line':
            
                ax.plot(xar, yar, color=color,label =keys_d[plot_colr], **plot_kwargs)
            else:
                raise IOError()
            
            #===================================================================
            # format
            #===================================================================
            #chang eto log scale
            ax.set_xscale(xscale)
            
            if not xlims is None:
                ax.set_xlim(xlims)
            
            
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                # first row
                if row_key == row_keys[0]:
                    #last col
                    if col_key == col_keys[-1]:
                        ax.legend()
                    
                # first col
                if col_key == col_keys[0]:
                    ax.set_ylabel(ylabel)
                
                #last row
                if row_key == row_keys[-1]:
                    ax.set_xlabel(xlabel)
                
                
                ax.grid()
        

        

        
        #=======================================================================
        # wrap
        #=======================================================================
        
        fname = 'progres_%s_%s' % (title,self.longname)
                
        fname = fname.replace('=', '-').replace(' ','').replace('\'','')
        
        
        return self.output_fig(fig, fname=fname)
        
    
    

#===============================================================================
# runners--------
#===============================================================================
def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r0',
        overwrite=True,
        
        #=======================================================================
        # data files
        #=======================================================================
        catalog_fp='',
        
        #=======================================================================
        # parameters
        #=======================================================================
         
 
 
        #=======================================================================
        # plot control
        #=======================================================================
        transparent=False,
        
        #=======================================================================
        # debugging
        #=======================================================================
 
        
        **kwargs):
    
    with RasterAnalysis(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                        
                       bk_lib = {
                           'catalog':dict(catalog_fp=catalog_fp),
                           
                           },
                 **kwargs) as ses:
        
        #=======================================================================
        # compiling-----
        #=======================================================================
        ses.logger.warning('test')
        ses.runRastAnalysis()
        
        #=======================================================================
        # PLOTS------
        #=======================================================================
        dx_raw = ses.retrieve('catalog')
        hi_res=350 #max we're using for hyd is 300
        for plotName, dx, xlims, xscale in [
            ('full', dx_raw,(10, 1e4), 'log',), 
            #('hi_res',dx_raw.loc[dx_raw.index.get_level_values('resolution')<=hi_res, :], (10, hi_res), 'linear')
            ]:
            print('\n\n%s\n\n'%plotName)
            #===================================================================
            # vs Resolution-------
            #===================================================================
            #===================================================================
            coln='MEAN'
            ses.plot_vsResolution(coln=coln, ylabel='mean depth (m)', dx_raw=dx,xlims=xlims,xscale=xscale,
                                  title='%s vs. %s (%s)'%(coln, ses.resCn, plotName))
              
            coln='wetAreas'
            ses.plot_vsResolution(coln=coln, ylabel='wet area (m2)', dx_raw=dx,xlims=xlims,xscale=xscale,
                                  title='%s vs. %s (%s)'%(coln, ses.resCn, plotName))
            
        #===================================================================
        # value distributions----
        #===================================================================
        #couldnt get this to be useful w/o droppping zeros (which corrupts the results)
        #=======================================================================
        # ses.plot_rvalues(figsize=(12,12), plot_type='gaussian_kde', drop_zeros=False,  write_stats=False,
        #                 #hrange=(0,10),  xlims=(0,10), 
        #                 ylims=(0,0.1),
        #                  #linewidth=0.75,
        #                  )
        #=======================================================================
        
 
        #=======================================================================
        # ses.plot_rvalues(figsize=(16,12), plot_type='violin', drop_zeros=False,  write_stats=False,
        #                  grid=False,
        #                  #yscale='log',
        #                  sharex='all',
        #                 #hrange=(0,10),  xlims=(0,10), 
        #                 #ylims=(1e-1,10**3),
        #                  #linewidth=0.75,
        #                  )
        #=======================================================================
        #=======================================================================
        # bin_max=5
        # ses.plot_rvalues(figsize=(12,12), plot_type='hist', drop_zeros=False, density=True, debug_max_len=None, write_stats=False,
        #                  xlims=(0,bin_max),ylims=(0,.2),bins=bin_max*4, bin_lims=(0,bin_max),
        #                  )
        #=======================================================================

        
        out_dir = ses.out_dir
    return out_dir

def dev():
    return run(
        tag='dev',
        compiled_fp_d={
            'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\drlay_lib_rast1_dev_0426.pickle',
            'rstats_basic':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\rstats_rast1_dev_0426.pickle',
            'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\wetAreas_rast1_dev_0426.pickle',
            'rstats':r'C:\LS\10_OUT\2112_Agg\outs\rastAnaly\dev\20220426\working\rstats_rastAnaly_dev_0426.pickle',
            }
                
        )
    
def r1():
    return run(
        tag='r1',
 
        compiled_fp_d = {
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\rast\r1\20220426\working\drlay_lib_rast_r1_0426.pickle',
        'rstats_basic':r'C:\LS\10_OUT\2112_Agg\outs\rast\r1\20220426\working\rstats_basic_rast_r1_0426.pickle',
        'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\rast\r1\20220426\working\wetAreas_rast_r1_0426.pickle',
        #'rstats':r'C:\LS\10_OUT\2112_Agg\outs\rast_analy\r1\20220426\working\rstats_rast_analy_r1_0426.pickle',
            }
        )
        
def r2():
    return run(
        tag='r1',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hrast1\hrast_run_index.csv',
        compiled_fp_d = {
 
            }
        )
    
if __name__ == "__main__": 
    
    #dev()
    r2()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    