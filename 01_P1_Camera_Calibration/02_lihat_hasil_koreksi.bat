@echo off
title Live Undistortion Viewer (Original vs Undistorted)
echo ========================================================
echo   PENAMPIL HASIL KOREKSI DISTORSI LENSA (P1)
echo ========================================================
echo.
cd /d "D:\Downloads\PBL_Vision_Project\01_P1_Camera_Calibration"
"D:\Downloads\CameraCalibration\CameraCalibration\.venv\Scripts\python.exe" save_realtime_result.py
pause
