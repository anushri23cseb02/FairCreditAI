@echo off
REM Stop FairCredit AI. Data (MySQL volume, models, data folder) is kept.
cd /d "%~dp0"
docker compose down
