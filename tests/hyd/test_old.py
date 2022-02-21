'''
Created on Feb. 18, 2022

@author: cefect
'''
import unittest
import tempfile
import numpy as np
import pytest
print('pytest.__version__:' + pytest.__version__)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
 
from numpy.testing import assert_equal  
from numpy import array, int32, float32, int64

import pandas as pd
from pandas import Index, Int64Index, RangeIndex
from pandas.testing import assert_frame_equal, assert_series_equal

from qgis.core import QgsVectorLayer


from agg.hyd.scripts import Model as CalcSession
from agg.hyd.scripts import StudyArea as CalcStudyArea

from agg.hyd.scripts import vlay_get_fdf


#===============================================================================
#COMPONENETS-------
#===============================================================================
class Basic(unittest.TestCase): #session level tester
    prec=3
    write=False
    @classmethod
    def setUpClass(cls, bk_lib={}, compiled_fp_d={}):
        cls.ses = CalcSession(out_dir = tempfile.gettempdir(), proj_lib=cls.proj_lib,
                              bk_lib = bk_lib, compiled_fp_d=compiled_fp_d, 
                              overwrite=True, tag=cls.tag
                              )
 
    @classmethod
    def tearDownClass(cls):
        cls.ses.__exit__()
        
class StudyArea(Basic):
    studyArea = 'point'
    @classmethod
    def setUpClass(cls,  **kwargs):
        #setup the session
        super(StudyArea, cls).setUpClass(**kwargs)
        
        #setup the study area
        kwargs = {k:getattr(cls.ses, k) for k in ['tag', 'prec', 'trim', 'out_dir', 'overwrite']}
        cls.studyArea= CalcStudyArea(session=cls.ses, name=self.studyArea, **cls.proj_lib[self.studyArea], **kwargs)
        print('setup studyArea %s'%cls.studyArea.name)
       
    @classmethod 
    def tearDownClass(cls):
        super(StudyArea, cls).__exit__()
        cls.studyArea.__exit__()
        
        
class StudyAreaZ(Basic):
    
 
    def test_all(self):
        
        #=======================================================================
        # #loop on each
        #=======================================================================
        kwargs = {k:getattr(self.ses, k) for k in ['tag', 'prec', 'trim', 'out_dir', 'overwrite']}
        for studyAreaName, pars_d in self.proj_lib.items():
            
            #init the study area
            with CalcStudyArea(session=self.ses, name=studyAreaName, **pars_d) as studyArea:
                pass
                

        
#===============================================================================
# COMPONENETS: PROJECTS--------
#===============================================================================
class Project1(object): #point finv
    proj_lib =     {
        'point':{
          'EPSG': 2955, 
         'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0218.geojson', 
         'dem': 'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\dem_obwb_test_0218.tif', 
         'wd_dir': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd',
         #'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
            }, 
        'poly':{
          'EPSG': 2955, 
         'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0219_poly.geojson', 
         'dem': 'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\dem_obwb_test_0218.tif', 
         'wd_dir': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd',
         #'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
            }, 
            }
 
#===============================================================================
# COMPONENTS: DKEY--------
#===============================================================================
"""not so good to use these for unit tests... could trigger lots of unexpected intputs"""
class Dkey(Basic): #dkey based test methods
    dkey = None #overwrite with subclass
    compiled_fp_d = dict() #overwrite to load precompileds
    @classmethod
    def setUpClass(cls):
        assert isinstance(cls.dkey, str) 
        super(Dkey, cls).setUpClass(bk_lib={cls.dkey:cls.bk_lib}, compiled_fp_d=cls.compiled_fp_d)
        cls.result = cls.ses.retrieve(cls.dkey, write=cls.write) #calls build_finv_gridPoly then sa_get
        
class Dkey_finv_agg(Dkey):
    dkey = 'finv_agg_d'
 
    def test_data(self):
        res_d = self.result
        true_d = self.true_lib['test_data']
        
        for studyArea, vlay in res_d.items():

            self.assertTrue(isinstance(vlay, QgsVectorLayer))
            
            ser = vlay_get_fdf(vlay)['gid'].sort_values().reset_index(drop=True)
            
            true_index = true_d[studyArea]
            
            self.assertEqual(len(ser), len(true_index))
            assert_equal(ser.index, true_index)
            
            """
            self.tag
            ser.values
            ser.index
            """
 
class Get_finv_gridPoly(StudyArea):
    def test_data(self):
        finv_vlay = cls.ses.vlay_load()
    
#===============================================================================
# test cases--------
#===============================================================================
class Test_p1_finv_gridded(Project1, StudyArea):
    tag='Test_p1_finv_gridded'
    #bk_lib = dict(aggType='gridded', aggLevel=50)
    true_lib = {
        'test_data':{
            'test1':pd.Series(
                array([1, 2, 3, 4, 5, 6], dtype=int64), name='gid',
                index=RangeIndex(start=0, stop=6, step=1)                
                )}}

class Test_p1_finv_none(Project1, Dkey_finv_agg):
    write=False
    tag='Test_p1_finv_gridded'
    bk_lib = dict(aggType='none', aggLevel=None)
    true_lib = {
        'test_data':{
            'test1':pd.Series(
                array([10899, 10900, 10901, 10902, 10903, 10904, 10905, 10906, 10907,
                       10908, 11250, 11251, 11252, 11254, 14376, 14377, 14378, 23029,
                       23030, 23032, 23084, 23086, 23087, 23088, 23089], dtype=int64),
                index=RangeIndex(start=0, stop=25, step=1), name='gid',
                )},
        
            }
    

class Test_p2_finv_gridded(Project1, Get_finv_gridPoly):
    write=True
    tag='Test_p2_finv_gridded'
    bk_lib = dict(aggType='gridded', aggLevel=50)
    true_lib = {
        'test_data':{
            'test1':RangeIndex(start=0, stop=10, step=1),
            }}

class Test_p2_finv_none(Project1, Dkey_finv_agg):
    write=True
    tag='Test_p2_finv_none'
    bk_lib = dict(aggType='none', aggLevel=None)
    true_lib = {
        'test_data':{
            'test1':pd.Series(
                array([10899, 10900, 10901, 10902, 10903, 10904, 10905, 10906, 10907,
                       10908, 11250, 11251, 11252, 11254, 14376, 14377, 14378, 23029,
                       23030, 23032, 23084, 23086, 23087, 23088, 23089], dtype=int64),
                index=RangeIndex(start=0, stop=25, step=1), name='gid',
                )},
        
            }
class Test_p1_sampGeo(Project1, Basic):
    tag = 'Test_p1_sampGeo'
    input_fp = { #input files generated from precurser steps
        'Test_p1_finv_gridded':r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\sampGeo\finv_gPoly_50_test1Testp1finvgridded0219.geojson',
        }
        
    true_lib = {
        'test_data':{
            'test1':pd.Series(
                array([1, 2, 3, 4, 5, 6], dtype=int64), name='gid',
                index=RangeIndex(start=0, stop=6, step=1)                
                )}}
    
    def setUp(self):

        
        self.ses.build_sampGeo()
    
    def test_data(self):
   
            
        res_d = self.result
        true_d = self.true_lib['test_data']
        
        for studyArea, vlay in res_d.items():
            self.assertTrue(isinstance(vlay, QgsVectorLayer))
            vlay_get_fdf(vlay)
            
class Test_tvals(unittest.TestCase):
    tag = 'Test_tvals'
    
    def setUp(self):
        self.mindex = 0
    

class Test_get_rsamps1(Project1, StudyArea):
    tag='Test_get_rsamps1'
    #===========================================================================
    # @classmethod
    # def setUpClass(cls):
    #     super(Test_get_rsamps2, cls).setUpClass()
    #===========================================================================

    def test_points(self):
        res_df = self.studyArea.get_rsamps(method='points', prec=self.prec)
        
        """
        res_df.columns
        res_df.T.values
        res_df.index
        """
        true_df = pd.DataFrame(
            np.array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.2 , 0.  ,
                    0.08, 0.  , 0.  , 0.  , 0.  , 0.  , 0.35, 0.42, 0.26, 0.36, 0.  ,
                    0.  , 0.  , 0.12]], dtype=np.float32),
            columns=pd.Int64Index([10899, 10900, 10901, 10902, 10903, 10904, 10905, 10906, 10907, 10908, 11250, 11251,
                11252, 11254, 14376, 14377, 14378, 23029, 23030, 23032, 23084, 23086, 23087, 23088,
                23089],dtype='int64', name='id'),
            index = pd.Index(['wd_test_0218'], dtype='object'),            
            ).T
            
        assert_frame_equal(true_df, res_df, check_dtype=False)
            
        
class Test_get_rsamps2(Project1, StudyArea):
    tag='Test_get_rsamps2'
    
    index = pd.Int64Index([10899, 10900, 10901, 10902, 10903, 10904, 10905, 10906, 10907, 10908, 10910, 10911,
                    10912, 10913, 11250, 11251, 11252, 11254, 14376, 14377, 14378, 23029, 23030, 23032,
                    23034, 23035, 23036, 23084, 23085, 23086, 23087, 23088, 23089],
                   dtype='int64', name='id')
 
    def test_zonal_mean(self):
        res_df = self.studyArea.get_rsamps(method='zonal', prec=self.prec, zonal_stats=[2])
        
        """
        res_df.columns
        res_df.T.values
        res_df.index
        """
        true_df = pd.DataFrame(
            np.array([[0.04 , 0.   , 0.242, 0.033, 0.   , 0.   , 0.042, 0.   , 0.06 ,
                0.195, 0.215, 0.452, 0.29 , 0.556, 0.068, 0.113, 0.   , 0.   ,
                0.01 , 0.   , 0.   , 0.273, 0.343, 0.238, 0.568, 0.307, 0.322,
                0.338, 0.023, 0.045, 0.   , 0.07 , 0.093]], dtype=np.float32),
            columns=self.index,
            index = pd.Index(['wd_test_0218'], dtype='object'),            
            ).T
            
        assert_frame_equal(true_df, res_df, check_dtype=False)
        
    def test_zonal_max(self):
        res_df = self.studyArea.get_rsamps(method='zonal', prec=self.prec, zonal_stats=[6])
 
        true_df = pd.DataFrame(
            np.array([[0.04, 0.  , 0.29, 0.07, 0.  , 0.  , 0.08, 0.  , 0.08, 0.29, 0.5 ,
                    0.47, 0.44, 0.76, 0.22, 0.14, 0.  , 0.  , 0.01, 0.  , 0.  , 0.35,
                    0.52, 0.37, 0.75, 0.46, 0.42, 0.38, 0.06, 0.07, 0.  , 0.11, 0.12]],
                  dtype=np.float32),
            columns=self.index,
            index = pd.Index(['wd_test_0218'], dtype='object'),            
            ).T
            
        assert_frame_equal(true_df, res_df, check_dtype=False)

    def test_zonal_min(self):
        res_df = self.studyArea.get_rsamps(method='zonal', prec=self.prec, zonal_stats=[5])
 
        true_df = pd.DataFrame(
            np.array([[0.04, 0.  , 0.15, 0.  , 0.  , 0.  , 0.01, 0.  , 0.06, 0.05, 0.1 ,
                    0.43, 0.18, 0.33, 0.  , 0.08, 0.  , 0.  , 0.01, 0.  , 0.  , 0.22,
                    0.22, 0.17, 0.45, 0.25, 0.22, 0.24, 0.  , 0.01, 0.  , 0.07, 0.08]],
                  dtype=float32),
            columns=self.index,
            index = pd.Index(['wd_test_0218'], dtype='object'),            
            ).T
            
        assert_frame_equal(true_df, res_df, check_dtype=False)
        
#==============================================================================
# suties-------
#==============================================================================
 
def get_suite():
    test_cases = [
        #Test_p1_finv_gridded,
        #Test_p1_finv_none,
        #Test_p2_finv_gridded,
        #Test_p1_finv_none,
        #Test_p1_sampGeo,
        Test_get_rsamps1,
        #Test_get_rsamps2
        ]
    
    suite = unittest.TestSuite()
    
    #make suite from list of testCases
    for testCase in test_cases:
        suite.addTest(unittest.makeSuite(testCase))
         
    print('built suite w/ %i'%suite.countTestCases())
    return suite
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(get_suite())