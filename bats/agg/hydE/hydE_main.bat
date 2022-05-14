 
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat

REM execute
 
@echo on
 
python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydE\main.py -t pre_cvh -n hydE01 -i 8 -dsampStage pre -downSampling Average -aggType convexHulls -aggIters 5
 
ECHO finished 

pause

 


