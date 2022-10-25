call C:\LS\06_SOFT\conda\env\agg\activate.bat
cd %REPO%
echo on
call conda env export >conda_env.yml

pause