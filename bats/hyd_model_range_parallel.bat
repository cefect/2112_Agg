REM paramters
SET ids=(40,41,42, 43,44,45,46,47,48)
SET LAG=600
SET NAME=hyd5
 

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute (using ping command to add some lag between calls)
@echo on
FOR %%i IN %ids% DO (
start cmd /k python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -n %NAME%
ping 127.0.0.1 -n %LAG% > nul
)




 


