@echo off
TITLE CardioPulse AI - Windows Environment Installer
echo ==========================================================
echo   CardioPulse AI - Smart Digital Stethoscope Windows Setup
echo ==========================================================

echo [*] Installing Python Dependencies from requirements.txt...
C:\ProgramData\anaconda3\python.exe -m pip install -r requirements.txt

echo.
echo ==========================================================
echo   [+] Windows Setup Completed Successfully!
echo   To launch web dashboard: C:\ProgramData\anaconda3\python.exe app.py
echo ==========================================================
pause
