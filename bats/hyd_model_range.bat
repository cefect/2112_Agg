REM paramters
SET ids=(0,1, 2)
 
@echo off
REM setup pyqgis
call C:\LS\06_SOFT\OSGeo4w\3.9.5\pyqgis_39LTR_setup.bat

REM set the assocated projects
set PYTHONPATH=%PYTHONPATH%;C:\LS\09_REPOS\01_COMMON\coms;C:\LS\09_REPOS\02_JOBS\2112_Agg\cef

REM execute
@echo on
FOR %%i IN %ids% DO (
python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -d)


pause

 


