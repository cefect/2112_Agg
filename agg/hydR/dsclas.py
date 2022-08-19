'''
Created on Aug. 19, 2022

@author: cefect

classifying downsample type
'''


from agg.hydR.hydR_scripts import RastRun


class DsampClassifier(RastRun):
    
    pass



def run():
    with DsampClassifier() as ses:
        pass
    
        #=======================================================================
        # prep layers
        #=======================================================================
        #load fine DEM
        
        #build coarse DEM
        
        #load fine WSE
        
        #build fine delta. (useful for later)
            #check all above zero. 
            #fillna=0 
        
        #build coarse WSE
        
        #=======================================================================
        # compute classes
        #=======================================================================
            #build a mask for each class
            
        #dry-dry: max(delta) <=0
        
        #wet-wet: min(delta) >0
        
        #partials: max(delta)>0 AND min(delta)==0
            #check this is all remainers
            
            #wet-partials: mean(DEM)<mean(WSE)
            
            #dry-partials: mean(DEM)>mean(WSE)
            
        #combine masks
            
            
    
        

if __name__ == "__main__": 
    run()
    
