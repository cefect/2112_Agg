REM paramters
SET ids=(0,1, 2)
 
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute
@echo on
FOR %%i IN %ids% DO (
python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -d)


pause

 


