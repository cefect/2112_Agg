'''
Created on Mar. 18, 2023

@author: cefect

compare aggregated grids to coarse hyd model grids
'''

from hcomp.scripts import HydCompareSession


def ahr_aoi08_r32_0130_30():
    return run_compare(
        wse1_fp=r'c:\LS\10_IO\2112_Agg\ins\hyd\ahr\ahr_aoi08_r32_0130_30\ahr_aoi08_r04_1215-0030_wse.tif',
        wse2_fp=r'c:\LS\10_IO\2112_Agg\ins\hyd\ahr\ahr_aoi08_r32_0130_30\ahr_aoi08_r32_1221-0030_wse.tif',
        dem1_fp=r'c:\LS\10_IO\2112_Agg\ins\hyd\ahr\ahr_aoi08_r32_0130_30\dem005_r04_aoi08_1210.asc',
        )


def run_compare(
        wse1_fp=None,
        wse2_fp=None,
        dem1_fp=None,
        ):
    """run the comparison"""
    
    with HydCompareSession() as ses:
        pass