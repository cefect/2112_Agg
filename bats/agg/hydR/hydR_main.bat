
SET dsampStage=post

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat

REM execute
 
@echo on
 
python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydR\main.py -t %dsampStage% -n hr6 -i 7 -dsampStage %dsampStage%
 
ECHO finished 

pause

 


