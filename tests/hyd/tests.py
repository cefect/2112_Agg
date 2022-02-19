'''
Created on Feb. 18, 2022

@author: cefect
'''
import unittest
import tempfile
from qgis.core import QgsVectorLayer


from agg.hyd.scripts import Session as CalcSession
from agg.hyd.scripts import vlay_get_fdf


#===============================================================================
# component classes-------
#===============================================================================
class Basic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ses = CalcSession(out_dir = tempfile.gettempdir(), proj_lib=cls.proj_lib,
                              bk_lib = {cls.dkey:cls.bk_lib}, overwrite=True
                              )
        
        cls.result = cls.ses.retrieve(cls.dkey) #calls build_finv_gridPoly then sa_get
        
        
class Project1(object):
    proj_lib =     {
        'test1':{
          'EPSG': 2955, 
         'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0218.geojson', 
         'dem': 'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\dem_obwb_test_0218.tif', 
         'wd_dir': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd',
         #'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
            }, 
            }


class Dkey_finv_gPoly(Basic):
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
    true_lib = {
        'test_data':
            {200:32, 0:450},
        }
 
#==============================================================================
# suties-------
#==============================================================================
 
def get_suite():
    test_cases = [
        Test_p1_finvgpoly
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