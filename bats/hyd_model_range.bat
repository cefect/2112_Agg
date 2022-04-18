REM paramters
REM SET ids=(90,91,92)
SET START=76
SET COUNT=3

SET NAME=hyd6

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute
SET /a "END=%START%+%COUNT%"
@echo on
FOR /l %%i IN (%START%,1,%END%) DO (
python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -n %NAME% 
)

FOR /l %%i IN (%START%,1,%START%+%COUNT%) DO (
ECHO finished %NAME% %%i
)
pause

 


