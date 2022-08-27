'''
unit tests for downsample v2
'''
 
from hp.np import apply_blockwise_ufunc, apply_blockwise, dropna
from hp.rio import RioWrkr, write_array, load_array
from numpy import array, dtype
from tests2.conftest import validate_dict, src_dir, get_abs
import numpy as np
import pytest, copy, os, random
import rasterio as rio
xfail = pytest.mark.xfail