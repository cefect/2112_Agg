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
temp_dir = os.path.join(tempfile.gettempdir(), __name__, datetime.datetime.now().strftime('%Y%m%d%S'))
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    
from tests.agg3.data.toy import crs_default, bbox_default
 

#===============================================================================
# fixtures
#===============================================================================
@pytest.fixture(scope='function')
def init_kwargs(tmp_path,write,logger, test_name):
    return dict(
        out_dir=tmp_path, 
        tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
        #prec=prec,
        proj_name='test', #probably a better way to propagate through this key 
        run_name=test_name[:8].replace('_',''),
        
        relative=True, write=write, #avoid writing prep layers
        
        logger=logger, overwrite=True, logfile_duplicate=False,
        )
    

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