'''
Created on Sep. 10, 2022

@author: cefect
'''
import numpy as np
import numpy.ma as ma
import pandas as pd
from pandas.testing import assert_index_equal
import os, copy, datetime
idx= pd.IndexSlice

#from definitions import max_cores
#from multiprocessing import Pool

#from agg2.haz.coms import coldx_d, cm_int_d
from hp.pd import append_levels
 
#===============================================================================
# setup matplotlib----------
#===============================================================================
  
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
from agg2.coms import Agg2DAComs
from hp.plot import view


def now():
    return datetime.datetime.now()


class ExpoDASession(ExpoSession, Agg2DAComs):
 
    
 
    
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
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('lsampsK',  subdir=True,ext='.pkl', **kwargs)
        
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
    
    def get_dsc_stats2(self, raw_dx,
                       ufunc_d = {'expo':'sum', 'wd':'mean', 'wse':'mean'},
                       **kwargs):
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dscStats',  subdir=True,ext='.pkl', **kwargs)
        
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
    
 
    
    def get_dsc_stats1(self, raw_dx, 
                        ufunc_l=['mean', 'sum', 'count'],
                        **kwargs):
        """compute stats groupbed by dsc on a layer
        
        
        WARNING: counts on residual data sets are pretty useless
            because these count presence of real values
            and there are only real values when both the baseline and the test are present
        """
        
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dscStats',  subdir=True,ext='.pkl', **kwargs)
        
        if not np.array_equal(raw_dx.columns.names, ['layer', 'scale']):
            raise IOError('bad columns: %s'%raw_dx.columns.names)
        
        res_d = dict()
        for scale, gdx in raw_dx.groupby('scale', axis=1):
            gdf = gdx.droplevel('scale', axis=1)
            
            #compute total
            d = {sn:getattr(gdf.drop('dsc', axis=1, errors='ignore'), sn)() for sn in ufunc_l}
            tdf = pd.concat(d, axis=1, names=['metric']).iloc[0, :].rename('full').to_frame().T
            
 
            

            #zonal
            if 'dsc' in gdx.columns.unique('layer'):
 
                d = {sn:getattr(gdf.groupby('dsc') , sn)() for sn in ufunc_l}
                rdfi = pd.concat(d, axis=1, names=['metric']).droplevel('layer', axis=1)
                
                rdf1 = pd.concat([tdf, rdfi])
                
            else:
                rdf1 = tdf
                
            #===================================================================
            # wrap
            #===================================================================
            res_d[scale] = rdf1
            
        #=======================================================================
        # wrap
        #=======================================================================
        rdx = pd.concat(res_d, axis=1, names=['scale'])
        
         
        # fill zeros 
        #=======================================================================
        # for sn in rdx.columns.unique('metric'):
        #     
        #     if not sn=='mean':
        #         idxi = idx[:, sn]
        #         sdx = rdx.loc[:, idxi]
        #         if sdx.isna().any().any():
        #             #print(sn) 
        #             rdx.loc[:, idxi] = sdx.fillna(0.0)
        #=======================================================================
                    
        log.debug('finished on %s'%str(rdx.shape))
        return rdx
                
                
    
    def get_dsc_stats(self, raw_dx, 
                      ufunc_l=['mean', 'sum', 'count'], 
                      multi=False,
                      **kwargs):
        """calc major stats grouped by dsc"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, layname, write = self._func_setup('dscStats',  subdir=True,ext='.pkl', **kwargs)
 
        start = now()
        gcols = ('scale', 'method')
        res_d = dict()
        res_d2 = dict()
        for i, (gkeys, gdx0) in enumerate(raw_dx.groupby(level=gcols, axis=1)):
            
            keys_d = dict(zip(gcols, gkeys))
            log.info('%i on %s'%(i+1, keys_d)) 
            
            #===================================================================
            # compute total
            #===================================================================
            
            
            gdf = gdx0.droplevel(gcols, axis=1).drop('dsc', axis=1, errors='ignore')
 
            
            d = {sn:getattr(gdf, sn)() for sn in ufunc_l}
            fdx = pd.concat(d, axis=1, names=['metric']).stack().rename('full').to_frame(
                ).T.reorder_levels(['layer', 'metric'], axis=1).sort_index(axis=1)
                
 
            
            #===================================================================
            # compute zonal
            #===================================================================                
            if 'dsc' in gdx0.columns.unique('layer'):            

                grouper = gdx0.droplevel(gcols, axis=1).groupby('dsc')            
     
                if multi:
                    """this is way slower"""
                    with Pool(3) as pool:
                        sum_future = pool.apply_async(grouper.sum)
                        sum_future.wait()
                         
                        mean_future = pool.apply_async(grouper.mean)
                        mean_future.wait()
                         
                        count_future = pool.apply_async(grouper.count)
                        count_future.wait()
                         
                    d = {'sum':sum_future.get(), 'mean':mean_future.get(), 'count':count_future.get()}
                    
                    #===============================================================
                    # """this is the same"""
                    # pool = Pool(3)
                    # 
                    # d = {
                    #     'sum':pool.apply_async(grouper.sum).get(), 
                    #      'mean':pool.apply_async(grouper.mean).get(),
                    #       'count':pool.apply_async(grouper.count).get()}
                    # 
                    # pool.close()
                    # pool.join()
                    #===============================================================
                    
                    
                else:
                    d = {sn:getattr(grouper, sn)() for sn in ufunc_l}
                    
                rdx1 = pd.concat(d, axis=1, names=['metric']).reorder_levels(['layer', 'metric'], axis=1)
                
                #merge w/ full
                rdx2 = pd.concat([rdx1, fdx]) 
                
            else:
                rdx2 = fdx
 
            #===================================================================
            # wrap
            #===================================================================                 
            rdx2.columns = append_levels(rdx2.columns, keys_d)
            
            res_d[i] = rdx2
            
 
                
            
        #=======================================================================
        # wrap
        #=======================================================================
        rdx = pd.concat(list(res_d.values()), axis=1)
        
 
        # fill zeros
 
        for sn in rdx.columns.unique('metric'):
            
            if not sn=='mean':
                idxi = idx[:, :, :, sn]
                sdx = rdx.loc[:, idxi]
                if sdx.isna().any().any():
                    #print(sn) 
                    rdx.loc[:, idxi] = sdx.fillna(0.0)
                    
                #fix count type
                if sn=='count':
                    rdx.loc[:, idxi] = rdx.loc[:, idxi].astype(int)
 
        rdx = rdx.reorder_levels(list(raw_dx.columns.names) + ['metric'], axis=1).sort_index(sort_remaining=True, axis=1)
        
        log.info('finished on %s in %.2f secs'%(str(rdx.shape), (now()-start).total_seconds()))
        
        
        return rdx
    
    
    """
    view(rdx.T)
    """
            
 
 
    
    
    
    
    
    
    
