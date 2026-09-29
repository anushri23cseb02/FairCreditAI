@echo off
REM Verify a running deployment. Uses Python on this computer if present (standard library only);
REM otherwise runs the same checks inside the backend container.
cd /d "%~dp0"

where python >nul 2>&1
if not errorlevel 1 goto haspython
where py >nul 2>&1
if not errorlevel 1 goto haspy
goto nopython

:haspython
python scripts\verify_deployment.py %*
goto end

:haspy
py -3 scripts\verify_deployment.py %*
goto end

:nopython
echo No Python found on this computer - running the checks inside the backend container.
docker compose exec backend python scripts/verify_deployment.py --inside-container %*

:end
