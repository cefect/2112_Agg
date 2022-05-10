 
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat

REM execute
 
@echo on
 
python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydR\main.py -t wse -n hr5 -i 2
 
ECHO finished 

pause

 


