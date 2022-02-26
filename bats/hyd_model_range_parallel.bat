REM paramters
SET ids=(3,4,5,6,7,8)
SET LAG=5
 

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute (using ping command to add some lag between calls)
@echo on
FOR %%i IN %ids% DO (
start cmd /k python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -d
ping 127.0.0.1 -n %LAG% > nul
)


pause

 


