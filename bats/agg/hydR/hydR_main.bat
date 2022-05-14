
SET dsampStage=postFN
SET 

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat

REM execute
 
@echo on
 
python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydR\main.py -t postFN_nn -n hydR01 -i 8 -dsampStage postFN -downSampling nn
 
ECHO finished 

pause

 


