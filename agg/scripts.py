'''
Created on Jan. 17, 2022

@author: cefect
'''
import os, sys, datetime, gc, copy


import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from hp.oop import Basic, Error


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

class Session(Basic):
    
    def get_matrix_fig(self, #conveneince for getting a matrix plot with consistent object access
                       row_keys, #row labels for axis
                       col_keys, #column labels for axis
                       
                       fig_id=0,
                       figsize=None,
                        tight_layout=False,
                        constrained_layout=True,
                        
                        sharex=False, 
                         sharey='row',
                        
                       ):
        
        
        #=======================================================================
        # defautls
        #=======================================================================
        if figsize is None: figsize=self.figsize
        
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(row_keys, list)
        assert isinstance(col_keys, list)
        
        if fig_id in plt.get_fignums():
            plt.close()
        
        #=======================================================================
        # build figure
        #=======================================================================
        
        fig = plt.figure(fig_id,
            figsize=figsize,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout)
        
        # populate with subplots
        ax_ar = fig.subplots(nrows=len(row_keys), ncols=len(col_keys),
                             sharex=sharex, sharey=sharey,
                             )
        
        #convert to array
        if not isinstance(ax_ar, np.ndarray):
            assert len(row_keys)==len(col_keys)
            assert len(row_keys)==1
            
            ax_ar = np.array([ax_ar])
            
        
        #=======================================================================
        # convert to dictionary
        #=======================================================================
        ax_d = dict()
        for i, row_ar in enumerate(ax_ar.reshape(len(row_keys), len(col_keys))):
            ax_d[row_keys[i]]=dict()
            for j, ax in enumerate(row_ar.T):
                ax_d[row_keys[i]][col_keys[j]]=ax
                
            
 
            
        return fig, ax_d
    
    def output_fig(self, 
                   fig,
                   
                   #file controls
                   out_dir = None, overwrite=True, 
                   out_fp=None, #defaults to figure name w/ a date stamp
                   fname = None, #filename
                   
                   #figure write controls
                 fmt='svg', 
                  transparent=True, 
                  dpi = 150,
                  logger=None,
                  ):
        #======================================================================
        # defaults
        #======================================================================
        if out_dir is None: out_dir = self.out_dir
        if overwrite is None: overwrite = self.overwrite
        if logger is None: logger=self.logger
        log = logger.getChild('output_fig')
        
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        #=======================================================================
        # precheck
        #=======================================================================
        
        assert isinstance(fig, matplotlib.figure.Figure)
        log.debug('on %s'%fig)
        #======================================================================
        # output
        #======================================================================
        if out_fp is None:
            #file setup
            if fname is None:
                try:
                    fname = fig._suptitle.get_text()
                except:
                    fname = self.name
                    
                fname =str('%s_%s'%(fname, self.resname)).replace(' ','')
                
            out_fp = os.path.join(out_dir, '%s.%s'%(fname, fmt))
            
        if os.path.exists(out_fp): 
            assert overwrite
            os.remove(out_fp)

            
        #write the file
        try: 
            fig.savefig(out_fp, dpi = dpi, format = fmt, transparent=transparent)
            log.info('saved figure to file:   %s'%out_fp)
        except Exception as e:
            raise Error('failed to write figure to file w/ \n    %s'%e)
        
        return out_fp

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
            
                            
 
     