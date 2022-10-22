'''
Created on Oct. 22, 2022

@author: cefect

2d profile example of wd averaging
'''
import os, pathlib, itertools, logging, sys
import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy
idx = pd.IndexSlice
 
#===============================================================================
# setup matplotlib----------
#===============================================================================
output_format='svg'
usetex=False
if usetex:
    os.environ['PATH'] += R";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
 
cm = 1/2.54
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
    'figure.figsize':(10,4),
    'legend.title_fontsize':'large',
    'text.usetex':usetex,
    }.items():
        matplotlib.rcParams[k] = v
   
print('loaded matplotlib %s'%matplotlib.__version__)
#===============================================================================
# setup logger
#===============================================================================
logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )

logger = logging.getLogger()
#===============================================================================
# funcs
#===============================================================================
def get_wsh_vals(cnt=5, mask_zeros=False):
    """build set of aggregated grid values
    
    Parameters
    ----------
    cnt: int
        two power. count of set. number of aggregations
    """
    log = logger.getChild('wsh')
    #build the scales from this
    scales_l = [2**(i) for i in range(cnt)]
    smax = max(scales_l)
    #assert smax==2**cnt
    log.info(f'on {cnt} scales:\n    {scales_l}')
    
    """implied when we convert to pandas
    #get x domain
    x_ar = np.linspace(0,2**cnt,num=2**cnt, endpoint=False)"""
    
    #===========================================================================
    # get fine y
    #===========================================================================
 
    
    domain = smax*2
    perturb = 2**3
    ys1_ar = np.concatenate(
        [np.full(domain//2, 0), #dry
          np.linspace(0,1,num=perturb//2, endpoint=True),
          #np.full(domain//2, 1), #wet
          np.linspace(1,0,num=perturb//2, endpoint=True),
          np.full(domain//2-perturb, 0), #dry            
            ]
        )
    
    log.info(ys1_ar.shape)
    
    if mask_zeros:
        ys1_mar = ma.array(ys1_ar, mask=ys1_ar==0)
    else:
        ys1_mar = ma.array(ys1_ar, mask=False)
    #===========================================================================
    # build aggregates
    #===========================================================================
    d=dict()
    for i, s2 in enumerate(scales_l):
        log.info(f'aggregating {s2}')
        #split and aggregagte
        """using half scales because we mirror below"""
        try:
            ys2_ar = ys1_mar.reshape(len(ys1_mar)//s2, -1).mean(axis=1)
        except Exception as e:
            raise IOError(s2)
        
        #disaggregate
        
       
        d[s2] = scipy.ndimage.zoom(np.where(~ys2_ar.mask,ys2_ar.data,  np.nan), s2, order=0, mode='reflect',   grid_mode=True)
        
    d[1] = ys1_ar
    #===========================================================================
    # merge
    #===========================================================================
    log.info(f'built {len(d)}')
    
    df= pd.DataFrame(d)
    
    return df

    df.plot()
 
    
def plot_profile(dx):
    
    #===========================================================================
    # defaults
    #===========================================================================
    log = logger.getChild('plot_profile')
    cmap = plt.cm.get_cmap(name='magma')
    
    """
    dx.plot(colormap=cmap)
    """
 
    mdex=dx.columns
    map_d = {'col':'method'}
    keys_all_d = {k:mdex.unique(v).tolist() for k,v in map_d.items()}
    #===========================================================================
    # setup plot
    #===========================================================================
    fig = plt.figure( 
                constrained_layout=True, )
    
    ax_ar = fig.subplots(nrows=1, ncols=len(keys_all_d['col']), sharex='all', sharey='all')    
    
    ax_d = dict(zip(keys_all_d['col'], ax_ar))
    #===========================================================================
    # loop and plot
    #===========================================================================
    for gcol, gdx in dx.groupby(map_d['col'], axis=1):
        ax = ax_d[gcol]
        
        gdx.droplevel(map_d['col'], axis=1).plot(ax=ax, colormap=cmap)
        
        ax.set_title(gcol)
        
        
    log.info(f'finsihed')
    """
    plt.show()
    """
 
 
 
        
 
    
 
 

if __name__ == "__main__":
    df1 = get_wsh_vals()
    df2 = get_wsh_vals(mask_zeros=True)
    
    dx=pd.concat({'zeros':df1, 'nulls':df2}, axis=1, names=['method', 's2'])
    
    
    plot_profile(dx)
 
    
    print('finished')