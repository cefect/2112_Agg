'''
Created on Oct. 26, 2022

@author: cefect
'''


from aggF.scripts import AggSession1F

 
class Plot_rlDelta_xb(object):
    def plot_matrix_rlDelta_xb(self,serx):
        """matrix plot of all functions rlDelta vs. xb"""
        log = self.logger.getChild('p')
           
        
class Session_AggF(AggSession1F, Plot_rlDelta_xb):
    pass