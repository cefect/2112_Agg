 
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat

SET CAT_FP=C:\LS\10_OUT\2112_Agg\lib\hydR01\hydR01_run_index.csv

REM execute
 
@echo on
 
python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydR\main.py -t pre -n hydR01 -i 8 -dsampStage pre -catalog_fp %CAT_FP%
 
ECHO finished 

pause

 


