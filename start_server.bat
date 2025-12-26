@echo off
cd /d "%~dp0"
echo Starting EcoSonicNet backend (serving UI + API on http://localhost:5000)...
C:\Users\dream\AppData\Local\Programs\Python\Python39\python.exe -m backend.app
pause

