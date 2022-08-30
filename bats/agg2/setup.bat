REM generic setup ufor hyd.model runs
@echo off
REM setup pyqgis
call "C:\Users\cefect\.venv\QGIS 3.22.8\agg\python-qgis-ltr_3228_agg_activate.bat"

REM set the assocated projects
set PYTHONPATH=%PYTHONPATH%;C:\LS\09_REPOS\02_JOBS\2112_Agg\cef;C:\LS\09_REPOS\01_COMMON\coms