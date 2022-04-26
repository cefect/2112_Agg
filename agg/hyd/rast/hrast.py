'''
Created on Apr. 26, 2022

@author: cefect

small analysis to focus on rasters
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, sys
import qgis.core
from qgis.core import QgsRasterLayer
import pandas as pd
import numpy as np

np.random.seed(100)

from agg.hyd.hscripts import Model, StudyArea, view

start = datetime.datetime.now()
print('start at %s' % start)


class RastRun(Model):
    def __init__(self,
                 name='rast',
                 data_retrieve_hndls={},
                 **kwargs):
        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            'drlay_lib':{ #overwrites Model's method
                'compiled':lambda **kwargs:self.load_layer_lib(**kwargs),
                'build':lambda **kwargs: self.build_drlays2(**kwargs),
                },
            'rstats_basic':{ #overwrites Model's method
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_stats(**kwargs),
                },
            'wetAreas':{ #overwrites Model's method
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_wetAreas(**kwargs),
                },
                        
            }}
        
        super().__init__( 
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
    #===========================================================================
    # DATA construction-----------
    #===========================================================================
    def runDownsample(self):
        
        self.retrieve('drlay_lib')
        
        self.retrieve('rstats_basic')
        
        self.retrieve('wetAreas')
    
    def build_drlays2(self,
                     
                     #parameters [calc loop]
                     iters=3, #number of downsamples to perform
                     resolution_scale = 3, 
                     base_resolution=None, #resolution of raw data
                     
                     #parameters [get_drlay]. for non base_resolution
                     dsampStage='wse',downSampling='Average',
                     
                     #outputs
                     out_dir=None,
                     
                     dkey='drlay_lib',logger=None,write=None,**kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_drlays')
        if write is None: write=self.write
        assert dkey=='drlay_lib'
        
        #resolutions
        if base_resolution is None:
            from definitions import base_resolution
        
        assert not dsampStage=='none'
        assert not downSampling=='none'
        
        temp_dir = self.temp_dir #collect
        #=======================================================================
        # build iter loop
        #=======================================================================
        #[10, 30, 90]
        resolution_iters = [base_resolution*(resolution_scale)**i for i in range(iters)]
        #=======================================================================
        # retrive rasters per StudyArea
        #=======================================================================
        """leaving everything on the StudyArea to speed things up"""
        
        #execute
        log.info('constructing %i: %s'%(len(resolution_iters), resolution_iters))
        res_lib = dict()
        cnt=0
        for i, resolution in enumerate(resolution_iters):
            log.info('\n\n%i/%i at %i\n'%(i+1, len(resolution_iters), resolution))
            
            #handle parameters
            if i==0:
                """because get_drlay has expectations for the base
                could also skip the downsampling... but this is mroe consistent"""
                dStage, dSamp='none', 'none'
            else:
                dStage, dSamp=dsampStage, downSampling
                
            #reset temp_dir
            self.temp_dir = os.path.join(temp_dir, 'r%i'%resolution)
            if not os.path.exists(self.temp_dir):os.makedirs(self.temp_dir)
            
            
            #build the depth layer
            res_lib[resolution] = self.sa_get(meth='get_drlay', logger=log.getChild(str(i)), dkey=dkey, write=False,
                                resolution=resolution, base_resolution=base_resolution,
                                dsampStage=dStage, downSampling=dSamp,
                                 **kwargs)
            
            cnt+=len(res_lib[resolution])
 
        self.temp_dir = temp_dir #revert
        log.info('finished building %i'%cnt)
        #=======================================================================
        # handle layers----
        #=======================================================================
 
        if write:
            if out_dir is None: out_dir = os.path.join(self.wrk_dir, dkey)
            ofp_lib = dict()
            
            #write each to file
            for resolution, layer_d in res_lib.items():
                ofp_lib[resolution] = self.store_layer_d(layer_d, dkey, logger=log,
                                   write_pick=False, #need to write your own
                                   out_dir = os.path.join(out_dir, 'r%i'%resolution)
                                   )
                
            #write the pick
            self.ofp_d[dkey] = self.write_pick(ofp_lib,
                os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
            
        return res_lib
            
    def load_layer_lib(self,  # generic retrival for layer librarires
                  fp=None, dkey=None,
                  **kwargs):
        """not the most memory efficient..."""
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('load.%s' % dkey)
        assert dkey in ['drlay_lib'], dkey
        
        #=======================================================================
        # load the filepaths
        #=======================================================================
        fp_lib = self.load_pick(fp=fp, dkey=dkey)   
        
        #=======================================================================
        # # load layers
        #=======================================================================
        lay_lib = dict()
        cnt = 0
        for k0, d0 in fp_lib.items():
            lay_lib[k0] = dict()
            for k1, fp in d0.items(): #usualy StudyArea
     
                log.info('loading %s.%s from %s' % (k0, k1, fp))
                
 
                ext = os.path.splitext(os.path.basename(fp))[1]
                #===================================================================
                # vectors
                #===================================================================
                if ext in ['.gpkg', '.geojson']:
                
                    lay_lib[k0][k1] = self.vlay_load(fp, logger=log, 
                                                   #set_proj_crs=False, #these usually have different crs's
                                                           **kwargs)
                elif ext in ['.tif']:
                    lay_lib[k0][k1] = self.rlay_load(fp, logger=log, 
                                                   #set_proj_crs=False, #these usually have different crs's
                                                           **kwargs)
                else:
                    raise IOError('unrecognized filetype: %s'%ext)
                cnt+=1
        
        log.info('finished loading %i'%cnt)
        return lay_lib
    
    def build_stats(self, #calc the layer stats 
                    dkey='rstats_basic',
                    logger=None, 
                     **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_stats')
        assert dkey=='rstats_basic'
        
        #=======================================================================
        # execut ethe function on the stack
        #=======================================================================
        return self.calc_on_layers(
            func=lambda rlay, **kwargs:self.rlay_getstats(rlay, **kwargs), 
            logger=log, dkey=dkey, **kwargs)
        
    def build_wetAreas(self,
                    dkey='wetAreas',
                    logger=None, **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_wetAreas')
        assert dkey=='wetAreas'
        
        
        dx= self.retrieve('rstats_basic')
        
        #=======================================================================
        # define the function
        #=======================================================================
        def func(rlay, logger=None):
            
            #build a mask layer
            mask_rlay = self.mask_build(rlay, logger=logger, layname='%s_mask'%rlay.name())
            
            #tally all the 1s
            wet_cnt = self.rasterlayerstatistics(mask_rlay)['SUM']
            
            #multiply by cell size
            return {dkey:wet_cnt * self.rlay_get_resolution(mask_rlay)}
 
            
        #=======================================================================
        # execute on stack
        #=======================================================================
        return self.calc_on_layers(
            func=func, 
            logger=log, dkey=dkey, **kwargs)
        
        """
        view(dx)
        """
 
    
    def calc_on_layers(self,
                       #data
                       lay_lib=None,
                       
                       #parameters
                       func=lambda rlay, **kwargs:{},
                       
                       #writing
                       write=None, dkey=None,
                       
                       logger=None, **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('calcLayers')
        
        if lay_lib is None: lay_lib = self.retrieve('drlay_lib')
        
        if write is None: write=self.write
        
        #=======================================================================
        # loop and execute on each layer
        #=======================================================================
        log.info('on %i'%len(lay_lib))
        
        res_d=dict()
        for resolution, d0 in lay_lib.items():
            d = dict()
            for studyArea, rlay in d0.items():
                #setup and precheck
                tagi = '%i.%s'%(resolution, studyArea)
                assert isinstance(rlay, QgsRasterLayer), tagi
                
                #match the layers crs
                self.qproj.setCrs(rlay.crs())
                
                #execute
                res = func(rlay, logger=log.getChild(tagi), **kwargs)
                
                #post
                assert isinstance(res, dict)                
                d[studyArea]=res
                
            #wrap
            res_d[resolution] = pd.DataFrame(d).T
            
        #=======================================================================
        # wrap
        #=======================================================================
        rdx = pd.concat(res_d, names=['resolution', 'studyArea'])
        
        assert isinstance(rdx, pd.DataFrame)
        
        log.info('finished on %i'%len(rdx))
        
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        
        return rdx
                
    #===========================================================================
    # visualize----------
    #===========================================================================


#===============================================================================
# class StudyArea2(RastRun, StudyArea):
#     pass
#===============================================================================

def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='tag',
        name='rast1',
        overwrite=True,
        trim=False,
        
        #=======================================================================
        # write control
        #=======================================================================
        write=True,
        exit_summary=True,
        write_lib=True, #enter the results into the library
        write_summary=True, #write the summary sheet
        #=======================================================================
        # #data
        #=======================================================================
        studyArea_l = None, #convenience filtering of proj_lib
        proj_lib = None,
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

 
        #=======================================================================
        # #parameters
        #=======================================================================
 
        #raster downSampling and selection  (StudyArea.get_raster())
        dsampStage='wse', downSampling='Average', severity = 'hi', 
        #resolution=5, this is what we iterate on
        
 
        
        #=======================================================================
        # meta
        #=======================================================================
 
        
 
        **kwargs):
    print('START run w/ %s.%s and '%(name, tag))
 
    #===========================================================================
    # study area filtering
    #===========================================================================
    if proj_lib is None:
        from definitions import proj_lib
    
    if not studyArea_l is None:
        print('filtering studyarea to %i: %s'%(len(studyArea_l), studyArea_l))
        miss_l = set(studyArea_l).difference(proj_lib.keys())
        assert len(miss_l)==0, 'passed %i studyAreas not in proj_lib: %s'%(len(miss_l), miss_l)
        proj_lib = {k:v for k,v in proj_lib.items() if k in studyArea_l}
    #===========================================================================
    # execute
    #===========================================================================
    with RastRun(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                     write=write,exit_summary=exit_summary,prec=prec,
                 bk_lib = {
 
                     
                     'drlay_d':dict( severity=severity, downSampling=downSampling, dsampStage=dsampStage),
 
                                          
                     },
                 **kwargs) as ses:
        
        ses.runDownsample()
        
 
        
 
        
    print('\nfinished %s'%tag)
    
    return 

def dev():
    return run(
        trim=True,
        tag='dev',
        compiled_fp_d={
            'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\drlay_lib_rast1_dev_0426.pickle',
            'rstats_basic':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\rstats_rast1_dev_0426.pickle',
            'wetAreas':r'C:\LS\10_OUT\2112_Agg\outs\rast1\dev\20220426\working\wetAreas_rast1_dev_0426.pickle',
            }
        )


if __name__ == "__main__": 
    
    dev()
    pass

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
 