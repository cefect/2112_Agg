SET DS_L=(pre, post, preGW, postFN)

SET LAG=5


REM execute
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat
@echo on
FOR %%i IN %DS_L% DO (
start cmd /k python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydR\main.py -t %%i -n hr8 -i 10 -dsampStage %%i
ping 127.0.0.1 -n %LAG% > nul
)
ECHO finished 

pause

 


