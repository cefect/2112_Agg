'''
Created on Feb. 24, 2022

@author: cefect

executing hyd.model from command line
'''
import sys, argparse
from agg.hyd.model import run_autoPars

if __name__ == "__main__":
    print(sys.argv)
    #===========================================================================
    # setup argument parser 
    #===========================================================================
    parser = argparse.ArgumentParser(description='execute hyd.models')
    #add arguments
    parser.add_argument("-modelID",'-id', help='specify the model code', default = 0, type=int)
    parser.add_argument("-tag",'-t', help='tag for the run', default = 'someTag', type=str)
    
    
    args = parser.parse_args()
    
    print('parser got these vars: %s'%vars(args)) #print all the parsed arguments in dictionary form
    
    run_autoPars(**vars(args))