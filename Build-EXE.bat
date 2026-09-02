@echo off
REM One-click portable build: produces dist\TLHEngine\TLHEngine.exe
cd /d "%~dp0"
python setup.py build_exe --clean
if errorlevel 1 (
  echo.
  echo Build failed. See messages above.
  pause
  exit /b 1
)
echo.
echo Done: dist\TLHEngine\TLHEngine.exe   (copy the whole dist\TLHEngine folder to deploy)
pause
