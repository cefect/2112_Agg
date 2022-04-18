REM paramters
REM SET ids=(90,91,92)
SET ids=(77,78)

SET NAME=hyd6

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute
 
@echo on
FOR %%i IN %ids% DO (
python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -n %NAME%
)

 
ECHO finished %ids%

pause

 


