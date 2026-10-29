@echo off
title Kalibrasi Kamera Real-time (P1)
echo ========================================================
echo   KALIBRASI KAMERA REAL-TIME CHECKERBOARD (P1)
echo ========================================================
echo.
cd /d "D:\Downloads\PBL_Vision_Project\01_P1_Camera_Calibration"
"D:\Downloads\CameraCalibration\CameraCalibration\.venv\Scripts\python.exe" realtimeCameraCalibration.py
pause
