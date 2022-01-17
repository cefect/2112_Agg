'''
Created on Jan. 17, 2022

@author: cefect
'''
import os, sys, datetime, gc, copy


import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def get_ax(
        figNumber=0,
        figsize=(4,4),
        tight_layout=False,
        constrained_layout=True,
        ):
    
    if figNumber in plt.get_fignums():
        plt.close()
    
    fig = plt.figure(figNumber,figsize=figsize,
                tight_layout=tight_layout,constrained_layout=constrained_layout,
                )
            
    return fig.add_subplot(111)

class Vfunc(object):
    ycn = 'rl'
    xcn = 'wd'
    
    def __init__(self,
                 vid=0, 
                 model_id=None, model_name='model_name',
                 name='vfunc',
                 logger=None,
                 session=None,
                 meta_d = {}, #optional attributes to attach
                 prec=4, #precsion for general rounding
                 ):
        
        #=======================================================================
        # attachments
        #=======================================================================
        self.model_id=model_id
        self.model_name=model_name
        self.vid=vid
        self.name=name
        self.logger = logger.getChild(name)
        self.session=session
        self.meta_d=meta_d
        self.prec=prec
        
        
    def set_ddf(self,
                df_raw,
                logger=None,
                ):
        
        #=======================================================================
        # defai;ts
        #=======================================================================
        if logger is None: logger=self.logger
        #log=logger.getChild('set_ddf')
        
        ycn = self.ycn
        xcn = self.xcn
        
        
        
        
        
        #precheck
        assert isinstance(df_raw, pd.DataFrame)
        l = set(df_raw.columns).symmetric_difference([ycn, xcn])
        assert len(l)==0, l
        
        
        df1 = df_raw.copy().sort_values(xcn).loc[:, [xcn, ycn]]
        
        #check monotoniciy
        for coln, row in df1.items():
            assert np.all(np.diff(row.values)>=0), '%s got non mono-tonic %s vals \n %s'%(
                self.name, coln, df1)
        
        self.ddf = df1
        self.dd_ar = df1.T.values
        #self.ddf_big = df1.copy()
        
        
        #log.info('attached %s'%str(df1.shape))
        
        return self
    
    def get_rloss(self,
                  xar,
                  prec=2,
                  #from_cache=True,
                  ):
        if prec is None: prec=self.prec
        assert isinstance(xar, np.ndarray)
        dd_ar = self.dd_ar
        #clean i tups
        """using a frame to preserve order and resolution"""
        rdf = pd.Series(xar, name='wd_raw').to_frame().sort_values('wd_raw')
        rdf['wd_round'] = rdf['wd_raw'].round(prec)
        

        
 
        
        #=======================================================================
        # identify xvalues to interp
        #=======================================================================
        #get unique values
        xuq = rdf['wd_round'].unique()
        """only interploating for unique values"""
        
        #check with unique values really need to be calculated
        #=======================================================================
        # if from_cache:
        #     bool_ar = np.invert(np.isin(xuq, self.ddf_big[self.xcn].values))
        # else:
        #     bool_ar = np.full(len(xuq), True)
        #=======================================================================
        
        #filter thoe out of bounds
        
        bool_ar = np.full(len(xuq), True)
        xuq1 = xuq[bool_ar]
        #=======================================================================
        # interploate
        #=======================================================================
        dep_ar, dmg_ar = dd_ar[0], dd_ar[1]
        #=======================================================================
        # def get_dmg(x):
        #     return np.interp(x, 
        #=======================================================================
        
        res_ar = np.apply_along_axis(lambda x:np.interp(x,
                                    dep_ar, #depths (xcoords)
                                    dmg_ar, #damages (ycoords)
                                    left=0, #depth below range
                                    right=max(dmg_ar), #depth above range
                                    ),
                            0, xuq1)
        
 
        #=======================================================================
        # plug back in
        #=======================================================================
        """may be slower.. but easier to use pandas for the left join here"""
        rdf = rdf.join(
            pd.Series(res_ar, index=xuq1, name=self.ycn), on='wd_round')
        
        """"
        
        ax =  self.plot()
        
        ax.scatter(rdf['wd_raw'], rdf['rl'], color='black', s=5, marker='x')
        """
        res_ar2 = rdf.sort_index()[self.ycn].values
 
 
        return res_ar2
    
    
    
    
    def plot(self,
             ax=None,
             figNumber=0,
             label=None,
             lineKwargs={},
             add_zeros=True,
             logger=None,
             ):
        
        #=======================================================================
        # defautls
        #=======================================================================
        if label is None: label=self.name
        #setup plot
        if ax is None:
            ax = get_ax(figNumber=figNumber)
            
        #get data
        ddf = self.ddf.copy()
        
        #add some dummy zeros
        if add_zeros:
            ddf = ddf.append(pd.Series(0, index=ddf.columns), ignore_index=True).sort_values(self.xcn)
            
        xar, yar = ddf.T.values[0], ddf.T.values[1]
        """
        plt.show()
        """
        ax.plot(xar, yar, label=label, **lineKwargs)
        
        return ax
            
                            
 
     