'''
Created on Feb. 18, 2022

@author: cefect
'''
import unittest
import tempfile
import numpy as np
import pandas as pd
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
from qgis.core import QgsVectorLayer


from agg.hyd.scripts import Session as CalcSession
from agg.hyd.scripts import StudyArea as CalcStudyArea

from agg.hyd.scripts import vlay_get_fdf


#===============================================================================
#COMPONENETS-------
#===============================================================================
class Basic(unittest.TestCase): #session level tester
    prec=3
    @classmethod
    def setUpClass(cls, bk_lib={}):
        cls.ses = CalcSession(out_dir = tempfile.gettempdir(), proj_lib=cls.proj_lib,
                              bk_lib = bk_lib, overwrite=True, tag=cls.tag
                              )
 
    @classmethod
    def tearDownClass(cls):
        cls.ses.__exit__()
        
class StudyArea(Basic):
    @classmethod
    def setUpClass(cls, name='test1', **kwargs):
        #setup the session
        super(StudyArea, cls).setUpClass(**kwargs)
        
        #setup the study area
        kwargs = {k:getattr(cls.ses, k) for k in ['tag', 'prec', 'trim', 'out_dir', 'overwrite']}
        cls.studyArea= CalcStudyArea(session=cls.ses, name=name, **cls.proj_lib[name], **kwargs)
        print('setup studyArea %s'%cls.studyArea.name)

        
#===============================================================================
# COMPONENETS: PROJECTS--------
#===============================================================================
class Project1(object): #point finv
    proj_lib =     {
        'test1':{
          'EPSG': 2955, 
         'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0218.geojson', 
         'dem': 'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\dem_obwb_test_0218.tif', 
         'wd_dir': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd',
         #'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
            }, 
            }
class Project2(object): #poly finv
    proj_lib =     {
        'test1':{
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
class Dkey(Basic): #dkey based test methods
    dkey = None #overwrite with subclass
    
    @classmethod
    def setUpClass(cls):
        assert isinstance(cls.dkey, str) 
        super(Dkey, cls).setUpClass(bk_lib={cls.dkey:cls.bk_lib})
        cls.result = cls.ses.retrieve(cls.dkey) #calls build_finv_gridPoly then sa_get
        
class Dkey_finv_gPoly(Dkey):
    dkey = 'finv_gPoly'
    
    bk_lib = dict(grid_sizes=[200])

        
    def test_keys(self):
        res_d = self.result
        
        #check grid size keys
        for studyArea, g_v_d in res_d.items():
            miss_l = set(g_v_d.keys()).symmetric_difference(self.bk_lib['grid_sizes'] + [0])
        
            self.assertEqual(len(miss_l), 0, msg='bad grid_sizes keys: %s'%miss_l)
 
                
    def test_data(self):
        res_d = self.result
        true_d = self.true_lib['test_data']
 
        
        for studyArea, g_v_d in res_d.items():
            for grid_size, vlay in g_v_d.items():
                true_v = true_d[grid_size]
                self.assertTrue(isinstance(vlay, QgsVectorLayer))
                
                df = vlay_get_fdf(vlay)
                
                #check grid_size
                self.assertEqual(len(df['grid_size'].unique()), 1)
                
                #asset count
                self.assertEqual(len(df), true_v)
                

                


#===============================================================================
# test cases--------
#===============================================================================
class Test_p1_finvgpoly(Project1, Dkey_finv_gPoly):
    tag='Test_p1_finvgpoly'
    true_lib = {
        'test_data':
            {200:32, 0:450},
        }
 
class Test_get_rsamps1(Project1, StudyArea):
    tag='Test_get_rsamps1'
    #===========================================================================
    # @classmethod
    # def setUpClass(cls):
    #     super(Test_get_rsamps2, cls).setUpClass()
    #===========================================================================

    def test_zonal(self):
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
            
        
        
        
        
        
#==============================================================================
# suties-------
#==============================================================================
 
def get_suite():
    test_cases = [
        #Test_p1_finvgpoly,
        Test_get_rsamps1
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