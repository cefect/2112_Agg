'''
Created on Sep. 26, 2022

@author: cefect
'''
import os, pathlib, itertools, logging, sys
import pandas as pd
import numpy as np
from hp.basic import get_dict_str, today_str, lib_iter
from hp.pd import append_levels, view
import matplotlib.pyplot as plt
import matplotlib
idx = pd.IndexSlice

res_fp_lib = {'r10':
              {
            'haz': r'C:\LS\10_OUT\2112_Agg\outs\agg2\r10\SJ\da\haz\20220926\SJ_r10_direct_0926_dprep.pkl',
            'exp':r'C:\LS\10_OUT\2112_Agg\outs\agg2\r8\da\20220926\bstats\SJ_r8_expo_da_0926_bstats.pkl'}
              }

 

def SJ_da_run(
        run_name='r10',
        **kwargs):    
    
    return run_plots_combine(res_fp_lib[run_name], proj_name='SJ', run_name=run_name, **kwargs)

def run_plots_combine(fp_lib,pick_fp=None,write=True,**kwargs):
    """da and figures which combine hazard and exposure
    
    
    
    """
    from agg2.expo.da import ExpoDASession as Session
 
    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    out_dir = os.path.join(pathlib.Path(os.path.dirname(fp_lib['exp'])).parents[1],'da', 'expo',today_str)
    print('out_dir:   %s'%out_dir)
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, write=write,**kwargs) as ses:
        log = ses.logger 
        
        if pick_fp is None:
            dx1 = ses.build_combined(fp_lib)
        else:
            dx1 = pd.read_pickle(pick_fp)
        #=======================================================================
        # separate data------
        #=======================================================================
        """index slicer for each axis"""
        mdex=dx1.columns
        
        #['direct_haz', 'direct_exp', 'filter_haz', 'filter_exp']
        col_keys = ['_'.join(e) for e in list(itertools.product(mdex.unique('method').values, ['haz', 'exp']))]
        row_keys = ['wd_bias', 'wse_error', 'exp_area', 'vol']
 
        #empty container
        idx_d = dict()
        post_lib=dict()
        
 
        
        def set_row(method, phase):
            coln = f'{method}_{phase}'
            assert not coln in idx_d
            idx_d[coln] = dict()
            post_lib[coln] = dict()
            return coln
        #=======================================================================
        # #direct_haz
        #=======================================================================
        ea_base = 's12AN'
        method, phase = 'direct', 'haz'
        
        coln = set_row(method, phase)        
        idx_d[coln]['wd_bias'] =    idx[phase, 's12N', method,'wd', 'mean', :]
        idx_d[coln]['wse_error'] =  idx[phase, 's12', method,'wse', 'mean', :]
        idx_d[coln]['exp_area'] =   idx[phase, ea_base, method,'wse', 'real_area', :]
        idx_d[coln]['vol'] =        idx[phase, 's12AN', method,'wd', 'vol', :]        
 
        #=======================================================================
        # direct_exp
        #=======================================================================
        method, phase = 'direct', 'exp'
        
        coln = set_row(method, phase)
        idx_d[coln]['wd_bias'] =    idx[phase, 's12N', method,'wd', 'mean', :]
        idx_d[coln]['wse_error'] =  idx[phase, 's12', method,'wse', 'mean', :]
        #idx_d[coln]['exp_area']=    idx[phase, ea_base, method,'expo', 'sum', :]
        #idx_d[coln]['vol'] =        idx[phase, 's12N', method,'wd', 'vol', :]
        
        #=======================================================================
        # #filter_haz
        #=======================================================================
        method, phase = 'filter', 'haz'
        
        coln = set_row(method, phase)        
        idx_d[coln]['wd_bias'] =    idx[phase, 's12N', method,'wd', 'mean', :]
        idx_d[coln]['wse_error'] =  idx[phase, 's12', method,'wse', 'mean', :]
        idx_d[coln]['exp_area'] =    idx[phase, ea_base, method,'wse', 'real_area', :]
        idx_d[coln]['vol'] =        idx[phase, 's12AN', method,'wd', 'vol', :]    
        
        #=======================================================================
        # filter_exp
        #=======================================================================
        method, phase = 'filter', 'exp'
        
        coln = set_row(method, phase)
        idx_d[coln]['wd_bias'] =    idx[phase, 's12N', method,'wd', 'mean', :]
        idx_d[coln]['wse_error'] =  idx[phase, 's12', method,'wse', 'mean', :]
        #idx_d[coln]['exp_area'] =   idx[phase, ea_base, method,'expo', 'sum', :]
        #idx_d[coln]['vol'] =        idx[phase, 's12N', method,'wd', 'vol', :]
        
        #=======================================================================
        # check
        #=======================================================================
        cnt = 0
        for colk, d in idx_d.items():
            assert colk in col_keys, colk
            for rowk, idxi in d.items(): 
                assert rowk in row_keys, rowk
                assert len(dx1.loc[:, idxi])>0, f'bad on {rowk}.{colk}'
                
                cnt+=1
                
        log.info('built %i data selectors'%cnt)
        
        #=======================================================================
        # #collect
        #=======================================================================
        data_lib = {c:dict() for c in row_keys} #matching convention of get_matrix_fig() {row_key:{col_key:ax}}
        
        for colk in col_keys:
            for rowk in row_keys:
                if rowk in idx_d[colk]:
                    idxi = idx_d[colk][rowk]
                    data_lib[rowk][colk] = dx1.loc[:, idxi].droplevel(list(range(5)), axis=1)                
            
            
        #===================================================================
        # plot------
        #===================================================================
        """such a custom plot no use in writing a function
        
        WSH:
            need to split axis
            direct
                why is normalized asset WD so high?
                    because assets are biased to dry zones
                    as the WD is smoothed, dry zones become wetter (and wet zones become drier)
        
        TODO:
        
        """
        dx1['exp'].loc[:, idx[('s1', 's12', 's2', 's12N'), 'direct', 'wd', :, 'full']]
        
        
        ax_d, keys_all_d = ses.plot_grid_d(data_lib, post_lib)
 
        
        
        ylab_d={
            #===================================================================
            #   'wd_bias':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$', 
            # 'wse_error':r'$\overline{WSE_{s2}}-\overline{WSE_{s1}}$', 
            #   'exp_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
            #   'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$',
            #===================================================================
              }
        
        ax_title_d = {
            'direct_haz': 'full domain', 
            'direct_exp':'exposed domain',
             'filter_haz':'full domain',
             'filter_exp':'exposed domain',            
            }
        
        ax_ylims_d = {
              #=================================================================
              'wd_bias':(-10,10**2), 
              # 'wse_error':(-1,15), 
              # #'exp_area':(-1,20),
              # 'vol':(-0.2, .01),
              #=================================================================
            }
        
        ax_yprec_d = {
              'wd_bias':0, 
              'wse_error':1, 
              'exp_area':0,
              'vol':1,
            }
        
        
        for row_key, col_key, ax in lib_iter(ax_d):
            ax.grid()
            
            
            
            #first col
            if col_key == col_keys[0]:
                if row_key in ylab_d:
                    ax.set_ylabel(ylab_d[row_key])
                else:
                    ax.set_ylabel(row_key)
                    

                    
                digits = ax_yprec_d[row_key]
                """not working... settig on all
                ax.yaxis.set_major_formatter(lambda x,p:f'%.{digits}f'%x)
                #ax.get_yaxis().set_major_formatter(lambda x,p:'{0:.{1}}'.format(x, digits))"""
                
                #last row
                if row_key==row_keys[-1]:
                    pass
                
            #first row
            if row_key==row_keys[0]:
                ax.set_title(ax_title_d[col_key])
                #ax.set_yscale('log')
                
                #last col
                if col_key==col_keys[-1]:
                    ax.legend()
                
                
            #last row
            if row_key==row_keys[-1]:
                ax.set_xlabel('resolution (m)')
                
                if 'exp' in col_key:
                    ax.axis('off')
                    for txt in ax.texts:
                        txt.set_visible(False)
                    
            #all?    
            #===================================================================
            # if row_key in ax_ylims_d:                
            #     ax.set_ylim(ax_ylims_d[row_key])
            #===================================================================
 
        
        #add the titles
        fig = ax.figure
        fig.suptitle('direct', x=0.32)        
        fig.text(0.8, 0.98, 'filter and subtract', size=matplotlib.rcParams['figure.titlesize'], ha='center')
        #fig.suptitle('filter and subtract', x=0.8)
        #=======================================================================
        # output
        #=======================================================================
        ofp = os.path.join(ses.out_dir, f'{ses.fancy_name}_matrix_combined.svg')
        ses.output_fig(fig, ofp=ofp)
        """
        plt.show()
        """

if __name__ == "__main__":
    SJ_da_run()
    #SJ_combine_plots_0919()
        