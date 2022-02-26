'''
Created on Jan. 17, 2022

@author: cefect
'''
 
import os, sys, datetime, gc, copy


import os, datetime, math, pickle, copy, shutil
import pandas as pd
import numpy as np



from hp.oop import Basic, Session, Error
from hp.Q import Qproj
from hp.pd import view, get_bx_multiVal
from hp.plot import Plotr




class Session(Session, Qproj):
    
    vidnm = 'df_id' #indexer for damage functions
    ycn = 'rl'
    xcn = 'wd'
    
    def __init__(self, 
                 data_retrieve_hndls={},
                 prec=2,
                 write=True,
                 **kwargs):
    
        #add generic handles
        """issues with chid instaancing when .update() is used"""
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            'df_d':{ #csv dump of postgres vuln funcs
                #'compiled':lambda x:self.build_df_d(fp=x), #no compiled version
                'build':lambda **kwargs:self.build_df_d(**kwargs),
                },
            
            'vid_df':{#selected/cleaned vfunc data
                #'compiled'  #best to just make a fresh selection each time
                'build':lambda **kwargs:self.build_vid_df(**kwargs)
                
                },
            
            'vf_d':{#initiliaed vfuncs
                    #these cant be compiled
                'build':lambda **kwargs:self.build_vf_d(**kwargs)
                
                },
            'vfunc':{ #retrieve a single vfunc
                'build':lambda **kwargs:self.build_vfunc(**kwargs)
                }
            }}
        
        super().__init__(work_dir = r'C:\LS\10_OUT\2112_Agg',
                         data_retrieve_hndls=data_retrieve_hndls,prec=prec,
                         #init_plt_d=None, #dont initilize the plot child
                         **kwargs)
        
        self.write=write
                
    #===========================================================================
    # BUILDERS---------------
    #===========================================================================
    
    def build_df_d(self,
                fp=r'C:\LS\09_REPOS\02_JOBS\2112_Agg\figueiredo2018\cef\csv_dump.xls',
                dkey=None,
                ):
        
        assert dkey=='df_d'
        log = self.logger.getChild('build_df_d')
        df_d = pd.read_excel(fp, sheet_name=None)
        
        log.info('loaded %i pages from \n    %s\n    %s'%(
            len(df_d), fp, list(df_d.keys())))
        
        
        return df_d
    
    def build_vid_df(self, #select and build vfunc data
                      df_d = None,
                      
                      #run control
                      write_model_summary=False,
                      
                      #simple vid selection
                      vid_l = None,
                      vid_sample=None,
                      
                      #attribute selection control
                      
                     selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[1, 2, 3, 4, 6, 7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
                     
                     max_mod_cnt = 10, #maximum dfuncs to allow from a single model_id
                     
                     
                     #keynames
                     vidnm = None, #indexer for damage functions
                     dkey='vid_df',
                     ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_vid_df')
        assert dkey == 'vid_df'
 
        if vidnm is None: vidnm=self.vidnm
        meta_lib=dict()
        
        
        #=======================================================================
        # loaders
        #=======================================================================
        if df_d is None:
            df_d = self.retrieve('df_d')
        
        #=======================================================================
        # attach data
        #=======================================================================
        #get metadata for selection
        df1, meta_lib['join_summary'] = self.get_joins(df_d)
        
        log.info('joined %i tabs to get %s' % (len(meta_lib['join_summary']), str(df1.shape)))
        

        
        #=======================================================================
        # basic selection-----------
        #=======================================================================
 
        #specivic fids
        if not vid_l is None:
            #check none o the other selectors are apssed
            assert vid_sample is None
            df3= df1.loc[vid_l, :]
        
 
        #=======================================================================
        # attribute selection-------
        #=======================================================================
        else:
            #id valid df_ids
            bx = get_bx_multiVal(df1, selection_d, log=log)
            
            df2 = df1.loc[bx, :] #.drop(list(selection_d.keys()), axis=1)
            
            log.info('selected %i/%i damageFunctions'%(len(df2), len(df1)))
            
     
            
            #=======================================================================
            # #by maximum model_id count
            #=======================================================================
            if not max_mod_cnt is None:
                coln = 'model_id'
                df3 = None
                for k, gdf in df2.groupby(coln):
                    if len(gdf)>max_mod_cnt:
                        log.warning('%s=%s got %i models... trimming to %i'%(
                            coln, k, len(gdf), max_mod_cnt))
                        
                        gdf1= gdf.iloc[:max_mod_cnt, :]
                    else:
                        gdf1 = gdf.copy()
                        
                    if df3 is None:
                        df3 = gdf1
                    else:
                        df3 = df3.append(gdf1)
                        
                #wrap
                log.info('cleaned out %i %s'%(len(df2.groupby(coln)), coln))
    
                         
                        
            else:
                df3 = df2.copy()
            
 
            
        #=======================================================================
        # random sampling within
        #=======================================================================
                #random vids
        if not vid_sample is None:
            res_df= df3.sample(vid_sample)
        else:
            res_df = df3
        #=======================================================================
        # checks
        #=======================================================================
        
        #check each of these have  ['wd', 'relative_loss']
        for tabName in  ['wd', 'relative_loss']:
            l = set(res_df.index).difference(df_d[tabName][vidnm])
            
            if len(l)>0:
                raise Error('requesting %i/%i %s w/o requisite tables\n    %s'%(
                    len(l), len(res_df), l))
            
        #=======================================================================
        # get a model summary sheet
        #=======================================================================
        if write_model_summary:
            coln = 'model_id'
            mdf = df1.drop_duplicates(coln).set_index(coln)
            
            
            mdf1 = mdf.join(df1[coln].value_counts().rename('vid_cnt')).dropna(how='all', axis=0)
            
            out_fp = os.path.join(self.out_dir, '%smodel_summary.csv'%self.resname)
            mdf1.to_csv(out_fp, index=True)
            
            log.info('wrote model summary to file: %s'%out_fp)
            
 
        log.info('finished on %s \n    %s'%(str(res_df.shape), res_df['abbreviation'].value_counts().to_dict()))
        
        #=======================================================================
        # write
        #=======================================================================
        
        
        return res_df
        

    
    def build_vf_d(self, #construct the vulnerability functions
                     vid_df=None,
                     df_d = None,
                     
                     #vfuncs selection
 
 
                     #key names
                     dkey=None,
                     vidnm = None, #indexer for damage functions
                     
                     ):
        """
        leaving this as a normal function
            dont want to load class objects from pickles
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('build_vf_d')
        assert dkey=='vf_d'
        
        if df_d is None: 
            df_d = self.retrieve('df_d')
            
        if vid_df is None: 
            vid_df=self.retrieve('vid_df')
            
 
        
            
        if vidnm is None: vidnm=self.vidnm
 
            
        
        #=======================================================================
        # spawn each df_id
        #=======================================================================
        def get_wContainer(tabName):
            df1 = df_d[tabName].copy()
            
            l = set(vid_df.index).difference(df1[vidnm])
            df1[vidnm].unique()
            if not len(l)==0:
                raise Error('missing %i keys in \'%s\''%(len(l), tabName))
            
 
            
            jdf = df_d[tabName+'_container'].copy()
            
            jcoln = jdf.columns[0]
            
            return df1.join(jdf.set_index(jcoln), on=jcoln) 
 
        lkp_d = {k:get_wContainer(k) for k in ['wd', 'relative_loss']}
 

        
        
        def get_by_vid(vid, k):
            try:
                df1 = lkp_d[k].groupby(vidnm).get_group(vid).drop(vidnm, axis=1).reset_index(drop=True)
            except Exception as e:
                raise Error(e)
            #drop indexers
            bxcol = df1.columns.str.contains('_id')
            return df1.loc[:, ~bxcol]
        
 
        vf_d = dict()
        log.info('spawning %i dfunvs'%len(vid_df))
        for i, (vid, row) in enumerate(vid_df.iterrows()):
            #log = self.logger.getChild('spawn_%i'%vid)
            #log.info('%i/%i'%(i, len(df2)))
            
            #===================================================================
            # #get dep-damage
            #===================================================================
            
            wdi_df = get_by_vid(vid, 'wd')
            rli_df = get_by_vid(vid, 'relative_loss')
            
            assert np.array_equal(wdi_df.index, rli_df.index)
            
            ddf =  wdi_df.join(rli_df).rename(columns={
                'wd_value':'wd', 'relative_loss_value':'rl'})
            #===================================================================
            # spawn
            #===================================================================
            vf_d[vid] = Vfunc(
                vid=vid, 
                model_id=row['model_id'],
                model_name=row['abbreviation'],
                name=row['abbreviation']+'_%03i'%vid,
                logger=self.logger, 
                session=self,
                meta_d = row.dropna().to_dict(), 
                ).set_ddf(ddf)
                
            #===================================================================
            # check
            #===================================================================
            for attn in ['xcn', 'ycn']:
                assert getattr(self, attn) == getattr(vf_d[vid], attn), attn
            
                
        log.info('spawned %i vfuncs'%len(vf_d))
        
        #self.vf_d = vf_d
        
        return vf_d
    
    
    def build_vfunc(self,
 
                    vid=1,
                    vf_d=None,
                    **kwargs):
        """
        not very nice... original setup was meant for bulk loading vfuncs
            while preserving the format of the csvs
        would be much nicer to pre-compile this into some database then just query the info needed
            (although I think this was the original format)
        """
        #=======================================================================
        # special linear testing func
        #=======================================================================
        if vid==0:
            max_depth=50
            ddf = pd.DataFrame({'rl':{0:0, 1:100},'wd':{0:0, 1:max_depth}})
 
            vf_d = {vid:Vfunc(
                vid=vid, 
                model_id=0,
                model_name='linear test',
                name='linear test',
                logger=self.logger, 
                session=self,
                meta_d = {}, 
                ).set_ddf(ddf)}
            
            
        #=======================================================================
        # retrieve from database
        #=======================================================================
        if vf_d is None:
            vid_df = self.build_vid_df(vid_l = [vid], **kwargs)
            
            vf_d = self.build_vf_d(dkey='vf_d', vid_df = vid_df)
 
        
        return vf_d[vid]
    
 
    
    #===========================================================================
    # HELPERS----------
    #===========================================================================
    
    def get_joins(self, df_d, 
                  vidnm=None):
        
        if vidnm is None: vidnm=self.vidnm
        
        
        df1 = df_d['damage_function'].set_index(vidnm)
        
 
        
        
        #join some tabs
        meta_d = dict()
        link_colns = set() #columns for just linking
        for tabName in [
                        'damage_format', 
                        'function_format', 
                        'coverage', 
                        'sector',
                        'unicede_occupancy', #property use type    
                        'construction_material',
                        'building_quality',
                        'number_of_floors',
                        'basement_occurance',
                        'precaution',
                        #'con',
                        
                        #'d',
                        ]:
            #===================================================================
            # #get id frame
            #===================================================================
            jdf = df_d[tabName].copy()
            
            #index
            assert vidnm == jdf.columns[0]

 
            if not jdf[vidnm].is_unique:
                raise Error('non-unique indexer \'%s\' on \'%s\''%(vidnm, tabName))
            
            jdf = jdf.set_index(vidnm)
            
            link_colns.update(jdf.columns)
            #===================================================================
            # #add descriptions
            #===================================================================
            container_tabName = '%s_container'%tabName
            assert container_tabName in df_d, container_tabName
            container_jdf = df_d[container_tabName]
            
            container_jcoln = container_jdf.columns[0]
            assert container_jcoln in jdf.columns
            
            jdf1 = jdf.join(container_jdf.set_index(container_jcoln), on=container_jcoln)
            
            
            #===================================================================
            # join to main
            #===================================================================
            

 
            df1 = df1.join(jdf1, on=vidnm)
            
            #===================================================================
            # meta
            #===================================================================
            meta_d[tabName] = {'shape':str(jdf1.shape), 'jcoln':vidnm, 'columns':jdf1.columns.tolist(),
                               'container_tabn':container_tabName, 
                               'desc_colns':container_jdf.columns.tolist(),
                               'link_colns':jdf.columns.tolist()}
        
        #=======================================================================
        # drop link columns
        #=======================================================================
        df2 = df1.drop(link_colns, axis=1)
        
        
        return df2, pd.DataFrame.from_dict(meta_d).T
    
    def _get_meta(self):
        d = super()._get_meta()
        
        return d
    
    


class Vfunc(object):
    ycn = 'rl'
    xcn = 'wd'
    
    def __init__(self,
                 vid=0, 
                 model_id=None, 
                 model_name='model_name',
                 name='vfunc',
                 logger=None,
                 session=None,
                 meta_d = {}, #optional attributes to attach
                 prec=4, #precsion for general rounding
                 relative=True, #whether this vfunc returns relative loss values (pct)
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
        self.relative=relative
        
        
    def set_ddf(self,
                df_raw,
                
                relative=None,
                
                #data correction
                set_loss_intercept=True,
                set_loss_zero=True,
                set_wd_indep=True,
                allow_rl_exceed=True, #for relative loss funtions, allow them to exceed 100
                
 
                logger=None,
                ):
        
        #=======================================================================
        # defai;ts
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('set_ddf')
        
        if relative is None: relative=self.relative
        
        ycn = self.ycn
        xcn = self.xcn
        
 
        #=======================================================================
        # #precheck
        #=======================================================================
        assert isinstance(df_raw, pd.DataFrame)
        l = set(df_raw.columns).symmetric_difference([ycn, xcn])
        assert len(l)==0, l
        
        
        
        #=======================================================================
        # clean
        #=======================================================================
        
        df1 = df_raw.copy().sort_values(xcn).loc[:, [xcn, ycn]]
        
        
        #=======================================================================
        # duplicate xvalues
        #=======================================================================
        """these violate the independence assumption"""
        if not df1[xcn].is_unique:
            bx = df1[xcn].duplicated(keep='first')
            msg = 'got %i duplicated xvals. set_wd_indep=%s'%(bx.sum(), set_wd_indep)
            if set_wd_indep:
                df1 = df1.loc[~bx, :]
            else:
                """these are not functions"""
                raise Error(msg)
        
        #=======================================================================
        # tables where rl[wd=0]=!0
        #=======================================================================
        if df1['wd'][0]==0:
            if df1.iloc[0, 1]>0:
                msg = 'rl[wd=0]!=0 and set_loss_zero=%s'%set_loss_zero
                if set_loss_zero:
                    df1.iloc[0,0] = 0.001
                    log.debug(msg + '  added 0.001 to wd')
                else:
                    log.warning(msg)
        
        #=======================================================================
        # tablees w/o wd=0
        #=======================================================================
        """because some tables dont extend to zero"""
        if df1['wd'].min()>0:
            msg = 'min(wd) >0 and set_loss_intercept=%s'%set_loss_intercept

            #add a dummy zerozero value for extrapolation
            if set_loss_intercept:
                df1 = df1.append(pd.Series(0.0, index=df1.columns), ignore_index=True).sort_values(xcn)
                log.debug(msg)
            else:
                log.warning(msg)
                
 
        #=======================================================================
        # post check
        #=======================================================================
        #check relative loss constraint
        if relative:
            """
            quite a few functions violate this
            """
            bx =  df1[ycn] > 100
            if bx.any():
                msg = 'got %i/%i RL values exceeding 100 (%s)'%(bx.sum(), len(bx), self.name)
                if not allow_rl_exceed:
                    raise Error(msg)
                else:
                    log.warning(msg)
            
        #check monotoniciy
        for coln, row in df1.items():
            assert np.all(np.diff(row.values)>=0), '%s got non mono-tonic %s vals \n %s'%(
                self.name, coln, df1)
        
        #=======================================================================
        # attach
        #=======================================================================
        self.ddf = df1
        self.dd_ar = df1.T.values
 
        
        return self
    
    def get_rloss(self,
                  xar,
                  prec=None, #precision for water depths
                  #from_cache=True,
                  ):
        """
        some functions like to return zero rloss
        
        """
        if prec is None: prec=self.prec
        assert isinstance(xar, np.ndarray)
        dd_ar = self.dd_ar.copy()
        
        #=======================================================================
        # prep xvals
        #=======================================================================
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
             set_title=True,
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
        
        if set_title:
            ax.set_title(self.name)
        
        return ax
            
class Catalog(object): #handling the simulation index and library
    df=None
    idn = 'modelID'
    

    
    def __init__(self, 
                 catalog_fp='fp', 
                 logger=None,
                 overwrite=True,
                 ):
        
        if os.path.exists(catalog_fp):
            self.df = pd.read_csv(catalog_fp)
            self.df_raw = self.df.copy() #for checking
        else:
            self.df_raw=pd.DataFrame()
        
        if logger is None:
            import logging
            logger = logging.getLogger()
        
        #=======================================================================
        # attachments
        #=======================================================================
        self.logger = logger.getChild('cat')
        self.overwrite=overwrite
        self.catalog_fp = catalog_fp
        
        #mandatory keys
        self.keys = [self.idn, 'name', 'tag', 'pick_fp', 'vlay_dir','pick_keys']
        
    def clean(self):
        raise Error('check consitency between index and library contents')
    
    def check(self):
        #=======================================================================
        # defai;lts
        #=======================================================================
        log = self.logger.getChild('check')
        df = self.df.copy()
        log.debug('on %s'%str(df.shape))
        #check columns
        miss_l = set(self.keys).difference(df.columns)
        assert len(miss_l)==0, miss_l
        
        assert df.columns.is_unique
        
        #check index
        assert df[self.idn].is_unique
        assert 'int' in df[self.idn].dtype.name
        
        #check filepaths
        errs_d = dict()
        for coln in ['pick_fp', 'vlay_dir']:
            assert df[coln].is_unique
            
            for id, path in df[coln].items():
                if not os.path.exists(path):
                    errs_d['%s_%i'%(coln, id)] = path
                    
        if len(errs_d)>0:
            raise Error('got %i/%i bad filepaths')
 
 
        
    
    def get(self):
        self.check()
        return self.df.set_index(self.idn).copy()
 
    def remove_model(self,
                        modelID,
                        df=None,
                        logger=None,
                        overwrite=None,
                        ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if df is None: df=self.df.copy()
        if logger is None: logger=self.logger
        log=logger.getChild('remove_model')
        if overwrite is None: overwrite=self.overwrite
        idn = self.idn
        
        
        bx = df[idn]==modelID
        if bx.any():
            log.warning('found %i/%i entries with matching modelID=%s'%(bx.sum(), len(bx), modelID))
            assert overwrite
            
            #delete records
            for fp in df.loc[bx, 'pick_fp']:
                log.info('removing %s'%fp)
                if os.path.exists(fp): 
                    os.remove(fp)
                
            for dirpath in df.loc[bx, 'vlay_dir']:
                log.info('removing all files from %s'%dirpath)
                if os.path.exists(dirpath): 
                    shutil.rmtree(dirpath)
                
            #delete index
            df1 = df.loc[~bx, :].copy()
 
        else:
            log.debug('no entry found')
            df1 = df.copy()
            
        #=======================================================================
        # set
        #=======================================================================
        self.df = df1
        
        return
    
    def add_entry(self,
                  cat_d):
        """
        cat_d.keys()
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log=self.logger.getChild('add_entry')
        cat_df = self.df
        

        #check
        """only allowing these columns for now"""
        miss_l = set(self.keys).difference(cat_d.keys())
        assert len(miss_l)==0, 'got %i unrecognized keys: %s'%(len(miss_l), miss_l)
        
        
        #=======================================================================
        # prepare new 
        #=======================================================================
        
        new_df = pd.Series(cat_d).to_frame().T
        
        #=======================================================================
        # append
        #=======================================================================
        if cat_df is None:
            cat_df = new_df
        else:
            cat_df = cat_df.append(new_df)
            
        #=======================================================================
        # wrap
        #=======================================================================
        self.df = cat_df
 
        
        log.info('added %i'%len(new_df))
                                
        

 
        
        
    def __enter__(self):
        return self
    def __exit__(self, *args, **kwargs):
        #=======================================================================
        # write if there was a change
        #=======================================================================
        if not np.array_equal(self.df, self.df_raw):
            log = self.logger.getChild('__exit__')
            df = self.df
            
            #===================================================================
            # delete the old for empties
            #===================================================================
            if len(df)==0:
                try:
                    os.remove(self.catalog_fp)
                    log.warning('got empty catalog... deleteing file')
                except Exception as e:
                    raise Error(e)
            else:
                #===============================================================
                # write the new
                #===============================================================
 
                df.to_csv(self.catalog_fp, mode='w', header = True, index=False)
                self.logger.info('wrote %s to %s'%(str(df.shape), self.catalog_fp))

               
 
     