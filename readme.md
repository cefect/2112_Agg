# Flood Hazard Grid Aggregation Scripts

[![DOI](https://zenodo.org/badge/440854458.svg)](https://zenodo.org/badge/latestdoi/440854458)

Work for main WRR manuscript aggregating grids, computing resample case, computing metrics, plotting.

## installation
build conda environment from ./environment.yml

create a ./definitions.py file similar to that shown below

add submodule
`git submodule add -b 2112_Agg https://github.com/cefect/coms.git`

add the new submodule to your python path

## USE

### figures

Figure 6. Bias from aggregation of four metrics
`agg2.run_da.SJ_da_run()`

Figure 5. Resample case classification progression for May 2018 Saint John River flood hazard data
`agg2.run_da.SJ_da_run()`

Figure 7. WSH difference maps for an example 512 m square region at five resolutions aggregated with the WSH Averaging routine
`agg2.run_da_maps.SJ_run()`

Figure S1. Full domain computation results. See main text for details
`agg2.haz.run_da.SJ_da_run()`

Figure S2. Exposed domain computation results. See main text for details
`agg2.expo.run_da.SJ_plots_0910()`




## Related Projects

[2210_AggFSyn](https://github.com/cefect/2210_AggFSyn.git):  Scripts for computing potential flood damage function error from aggregation against synthetically produced depths. ICFM9 work

[2112_AggAnal](https://github.com/cefect/2112_AggAnal): work for 2023 WRR aggregation supplement. copied from FloodGridPerformance (2023-06-30)

[FloodGridPerformance](https://github.com/cefect/FloodGridPerformance): simple tool for computing the performance of some flood grids (against observations). wrapper on fperf.

[2207_dscale2](https://github.com/cefect/2207_dscale2): project for generating analog inundation grids with LISFLOOD. 

[FloodPolisher](https://github.com/cefect/FloodPolisher): mid-2022 inundation downscaling work using simple growth. pyqgis. 

[FloodRescaler](https://github.com/cefect/FloodRescaler): public repo with simple QGIS tools included in Agg publication.  

[2112_agg_pub](https://github.com/cefect/2112_agg_pub): public repo of analysis for aggregation paper. 

[FloodDownscaler](https://github.com/cefect/FloodDownscaler): Work for manuscript to downscale inundation grids, compute performance against observations, and plot




### definitions.py
```
import os
proj_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef'
src_dir = proj_dir
src_name='agg'

logcfg_file=r'C:\LS\09_REPOS\01_COMMON\coms\logger.conf'

root_dir=r'C:\LS\10_IO\2112_Agg'
wrk_dir=root_dir

#path to latex engine
os.environ['PATH'] += R";C:\Program Files\MiKTeX\miktex\bin\x64"
```

 