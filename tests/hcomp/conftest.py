'''
Created on Mar. 18, 2023

@author: cefect
'''

import pytest, os, tempfile, datetime
import rasterio as rio
import shapely.geometry as sgeo
from shapely.geometry import mapping, Polygon
 
import fiona
import fiona.crs
from pyproj.crs import CRS


#===============================================================================
# defaults
#===============================================================================
temp_dir = os.path.join(tempfile.gettempdir(), __name__, datetime.datetime.now().strftime('%Y%m%d'))
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    
crs_default = CRS.from_user_input(25832)
bbox_default = sgeo.box(0, 0, 60, 90)

#===============================================================================
# helpers-------
#===============================================================================
def get_aoi_fp(bbox, crs=crs_default, ofp=None):
    
    if ofp is None:
        ofp = os.path.join(temp_dir, 'aoi.geojson')
        
    # write a vectorlayer from a single bounding box
    assert isinstance(bbox, Polygon)
    with fiona.open(ofp, 'w', driver='GeoJSON',
        crs=fiona.crs.from_epsg(crs.to_epsg()),
        schema={'geometry': 'Polygon',
                'properties': {'id':'int'},
            },
 
        ) as c:
        
        c.write({ 
            'geometry':mapping(bbox),
            'properties':{'id':0},
            })
        
    return ofp