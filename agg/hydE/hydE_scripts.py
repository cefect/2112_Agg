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
from qgis.core import QgsRasterLayer, QgsMapLayerStore
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal


idx = pd.IndexSlice
from hp.exceptions import Error
from hp.pd import get_bx_multiVal
import hp.gdal

from hp.Q import assert_rlay_equal
from agg.hyd.hscripts import Model, StudyArea, view, RasterCalc
from agg.hydR.hr_scripts import RastRun

class ExpoRun(RastRun):
    def __init__(self,
                 name='expo',
                 data_retrieve_hndls={},
 
                 **kwargs):
        

        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            #depth rasters
 
                        
            }}
        
        super().__init__( 
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
    def runExpo(self):
        
        self.retrieve('finv_agg_d')
        
    
 
        

