REM paramters
SET ids=(23, 24)
SET LAG=2
SET NAME=hyd4
 

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute (using ping command to add some lag between calls)
@echo on
FOR %%i IN %ids% DO (
start cmd /k python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -n %NAME%
ping 127.0.0.1 -n %LAG% > nul
)


pause

 


