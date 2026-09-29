@echo off
REM Start FairCredit AI on Windows. Safe to re-run; never deletes data.
REM Written without parenthesised blocks so folder names like "FairCreditAI_final (1)" work.
cd /d "%~dp0"

docker info >nul 2>&1
if errorlevel 1 goto nodocker

if exist .env goto haveenv
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\init_env.ps1
if errorlevel 1 goto envfail
:haveenv

docker compose up -d --build
if errorlevel 1 goto composefail

docker compose ps
echo.
echo Frontend : http://localhost:8501
echo API docs : http://localhost:8000/docs
echo Other computers on your network: http://THIS-COMPUTER-IP:8501  (see DEPLOYMENT.md)
echo Containers can take up to a minute to become healthy. Check with: verify.bat
goto end

:nodocker
echo Docker is not running. Open Docker Desktop, wait until it says it is running, then run start.bat again.
exit /b 1

:envfail
echo Could not create the .env file - see the message above.
exit /b 1

:composefail
echo docker compose failed - see the message above. Run: docker compose logs
exit /b 1

:end
