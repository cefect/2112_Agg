'''
Created on Sep. 9, 2022

@author: cefect
'''
import os, datetime

from hp.oop import Session, today_str

class Agg2Session(Session):
    def __init__(self,
                 case_name='SJ',
                 out_dir=None, fancy_name=None,
                 proj_name='agg2',
                 scen_name='direct', 
                 run_name='r1',
                 subdir=False,
                 wrk_dir=None,
                 engine='np', 
                 **kwargs):
        """handle project specific naming
        
        Parameters
        ----------
        case_name: str
            case study name
        
        scen_name: str
            scenario name (i.e., method)
            
        engine: str, default 'np'
            whether to use dask or numpy
 
        """
        if wrk_dir is None:
            from definitions import wrk_dir
            
        if out_dir is None:
            out_dir = os.path.join(wrk_dir, 'outs', proj_name, run_name, case_name,   scen_name, today_str)
            
        if fancy_name is None:
            fancy_name = '_'.join([case_name,run_name, scen_name, datetime.datetime.now().strftime('%m%d')])
        
        super().__init__( wrk_dir=wrk_dir, proj_name=proj_name,
                         out_dir=out_dir, fancy_name=fancy_name,subdir=subdir,
                         **kwargs)
        
        self.scen_name=scen_name
        self.engine=engine
        
 