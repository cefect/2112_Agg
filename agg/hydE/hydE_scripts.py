'''
Created on May 12, 2022

@author: cefect

scripts for running exposure calcs on hydR outputs

loop on studyarea (load all the rasters)
    loop on finv
        loop on raster
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, sys
import qgis.core
from qgis.core import QgsRasterLayer, QgsMapLayerStore, QgsWkbTypes
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal


idx = pd.IndexSlice
from hp.exceptions import Error
from hp.basic import set_info
from hp.pd import get_bx_multiVal
import hp.gdal

from hp.Q import assert_rlay_equal, vlay_get_fdf
from agg.hyd.hscripts import Model, StudyArea, view, RasterCalc
from agg.hydR.hr_scripts import RastRun

class ExpoRun(RastRun):
    ridn='rawid'
    def __init__(self,
                 name='expo',
                 data_retrieve_hndls={},
 
                 **kwargs):
        

        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            # aggregating inventories
            'finv_agg_d':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_layer_d(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_finv_agg2(**kwargs),
                },
 
                        
            }}
        
        super().__init__( 
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
    def runExpo(self):
        
        self.retrieve('finv_agg_d')
        
    def build_finv_agg2(self,  # build aggregated finvs
                       dkey=None,
                       
                       # control aggregated finv type 
                       aggType=None,
                       
                       #aggregation levels
                       aggLevel_l=None,
                       iters=3, #number of aggregations to perform
                       agg_scale=4,
                       
                       #defaults
                       proj_lib=None,
                       write=None, logger=None,**kwargs):
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
        if logger is None: logger=self.logger
        log = logger.getChild('build_finv_agg')
        if write is None: write=self.write
 
        assert dkey in ['finv_agg_d',
                        #'finv_agg_mindex', #makes specifycing keys tricky... 
                        ], 'bad dkey: \'%s\''%dkey
 
        gcn = self.gcn
        saCn=self.saCn
        log.info('building \'%s\' ' % (aggType))
        
        
        #=======================================================================
        # clean proj_lib
        #=======================================================================
        if proj_lib is None:
            proj_lib = copy.deepcopy(self.proj_lib)
        for sa, d in proj_lib.items():
            for k in ['wse_fp_d', 'dem_fp_d']:
                if k in d:
                    del d[k]
                    
        #=======================================================================
        # build aggLevels
        #=======================================================================
        #[1, 4, 16]
        if aggLevel_l is None:
            aggLevel_l = [(agg_scale)**i for i in range(iters)]
        
        assert max(aggLevel_l)<1e3
        
        #=======================================================================
        # build aggregated finvs------
        #=======================================================================
        """these should always be polygons"""
 
        res_d = self.sa_get(meth='get_finv_agg_d', write=False, dkey=dkey, get_lookup=True,
                             aggLevel_l=aggLevel_l,aggType=aggType, **kwargs)
        
 
        
        # unzip
        finv_gkey_df_d, finv_agg_d = dict(), dict()
        for studyArea, d in res_d.items():
            finv_gkey_df_d[studyArea], finv_agg_d[studyArea] = d['faMap_dx'], d['finv_d']
            
        assert len(finv_gkey_df_d) > 0, 'got no links!'
        assert len(finv_agg_d) > 0, 'got no layers!'

        
        
        #=======================================================================
        # check
        #=======================================================================
        #invert to match others
        dnew = {i:dict() for i in aggLevel_l}
        for studyArea, d in finv_agg_d.items():
            for aggLevel, vlay in d.items():
                """relaxing this
                assert vlay.wkbType()==3, 'requiring singleParts'"""
                assert 'Polygon' in QgsWkbTypes().displayString(vlay.wkbType())
                dnew[aggLevel][studyArea] = vlay
                
        finv_agg_d = dnew
        #=======================================================================
        # handle layers----
        #=======================================================================
        """saving write till here to match other functions
        might run into memory problems though....
        consider writing after each studyArea"""
        dkey1 = 'finv_agg_d'
        if write:
            self.store_lay_lib(dkey1, finv_agg_d, logger=log)
        
        self.data_d[dkey1] = finv_agg_d
        #=======================================================================
        # handle mindex-------
        #=======================================================================
        """creating a index that maps gridded assets (gid) to their children (id)
        seems nicer to store this as an index
        
        """
        #=======================================================================
        # assemble
        #=======================================================================
        
        dkey1 = 'finv_agg_mindex'
        dx = pd.concat(finv_gkey_df_d, verify_integrity=True, names=[self.saCn, self.ridn]) 
        
 
        #=======================================================================
        # check
        #=======================================================================
        #retrieve indexers from layer
        def func(vlay, logger=None, meta_d={}):
            df = vlay_get_fdf(vlay)
            assert len(df.columns)==1
            df1 = df.set_index(gcn).sort_index()
            df1.index.name=vlay.name()
            return df1           
        
        
        index_lib = self.calc_on_layers(lay_lib=finv_agg_d, func=func, subIndexer='aggLevel', 
                                  format='dict',
                            write=False, dkey=dkey1)
        
        d = dict()
        for k,v in index_lib.items():
            d[k] = pd.concat(v, names=[self.saCn, gcn])
            
        mindex = pd.concat(d, names=['aggLevel']).reorder_levels([self.saCn, 'aggLevel', gcn]).sort_index().index
 
        gb = mindex.to_frame().groupby(level=[saCn, 'aggLevel'])
        #check against previous
        for aggLevel, col in dx.items():
            for studyArea, gcol in col.groupby(level=self.saCn):
                layVals = gb.get_group((studyArea, aggLevel)).index.unique(gcn).values
                d = set_info(gcol.values, layVals, result='counts')
                if not d['symmetric_difference']==0:
                    raise Error('%s.%s \n    %s'%(aggLevel, studyArea, d))
 
            
 
 
        #=======================================================================
        # write
        #=======================================================================
        # save the pickle
        if write:
            self.ofp_d[dkey1] = self.write_pick(dx,
                           os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey1, self.longname)), logger=log)
        
        # save to data
        self.data_d[dkey1] = copy.deepcopy(dx)
 
        #=======================================================================
        # return requested data
        #=======================================================================
        """while we build two results here... we need to return the one specified by the user
        the other can still be retrieved from the data_d"""
 
        if dkey == 'finv_agg_d':
            result = finv_agg_d
        elif dkey == 'finv_agg_mindex':
            result = dx
        
        return result
        
    
 
        

