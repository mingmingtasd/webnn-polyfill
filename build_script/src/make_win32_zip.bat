@echo off

if "%1" == "" (
  echo usage: make_zip builddir
  echo.
  echo   builddir   build output directory ^(e.g., out\Release^)
  goto :eof
)

setlocal

echo %1
echo %2
set makezip=%1\chrome\tools\build\make_zip.py
echo %makezip%
set cfg=%2\FILES.cfg
set builddir=%1\out\win_x64_release
set archive=%1\out\win_x64_release\chrome-win32.zip

python %makezip% %builddir% %cfg% %archive%
