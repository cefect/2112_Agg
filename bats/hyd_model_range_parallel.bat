REM paramters
SET ids=(34,35,36,37)
REM ,)

SET LAG=180
SET NAME=hyd6

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute (using ping command to add some lag between calls)
@echo on
FOR %%i IN %ids% DO (
start cmd /k python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -n %NAME%
ping 127.0.0.1 -n %LAG% > nul
)




 


