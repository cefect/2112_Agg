:: fully remove a submodule
set SUBMOD=coms
git rm --cached -r %SUBMOD%
rmdir /S /Q .git\modules\%SUBMOD%
rmdir /S /Q %SUBMOD%
git config --remove-section submodule.%SUBMOD%

ECHO finished

cmd.exe /k