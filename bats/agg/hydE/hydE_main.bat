 
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat

SET CAT_FP=C:\LS\10_OUT\2112_Agg\lib\hydR02\hydR02_run_index_copy.csv
REM execute
 
@echo on
 
python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydE\main.py -t pre_cvh -n hydE02 -i 8 -dsampStage pre -downSampling Average -aggType convexHulls -aggIters 5 -catalog_fp %CAT_FP% -idcn 5
 
ECHO finished 

pause

 


