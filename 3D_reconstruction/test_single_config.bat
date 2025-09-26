@echo off
REM Test Single Configuration - Debug Version
setlocal EnableDelayedExpansion

set "DATASET_NAME=lego2"
set "CONFIG_NAME=original"
set "DATASET_DIR=datasets\%DATASET_NAME%"
set "RESULTS_DIR=%DATASET_DIR%\3dgs_results"
set "GAUSSIAN_SPLATTING_DIR=gaussian-splatting"
set "OUTPUT_DIR=%GAUSSIAN_SPLATTING_DIR%\output\%DATASET_NAME%_%CONFIG_NAME%"

echo ========================================
echo Testing Single Configuration: %CONFIG_NAME%
echo ========================================
echo Dataset: %DATASET_NAME%
echo Output: %OUTPUT_DIR%
echo ========================================

REM Check if we have a trained model
if not exist "%RESULTS_DIR%\weights\%CONFIG_NAME%" (
    echo ERROR: No trained model found for %CONFIG_NAME%
    echo Looking for: %RESULTS_DIR%\weights\%CONFIG_NAME%
    echo Available weights:
    dir "%RESULTS_DIR%\weights\" /b 2>nul
    pause
    exit /b 1
)

REM Copy trained model to temporary location for rendering
echo Copying trained model for rendering...
if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"
xcopy "%RESULTS_DIR%\weights\%CONFIG_NAME%" "%OUTPUT_DIR%" /e /i /q
echo Model copied successfully

REM Render test set only (200 images)
echo.
echo ========================================
echo Rendering Test Set (200 images)
echo ========================================
python "%GAUSSIAN_SPLATTING_DIR%\render.py" -m "%OUTPUT_DIR%" --skip_train --grayscale

if errorlevel 1 (
    echo ERROR: Rendering failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Checking Render Results
echo ========================================
echo Output directory structure:
dir "%OUTPUT_DIR%" /s /b

echo.
echo Test renders location:
if exist "%OUTPUT_DIR%\test\ours_7000\renders" (
    echo SUCCESS: Found renders at %OUTPUT_DIR%\test\ours_7000\renders
    dir "%OUTPUT_DIR%\test\ours_7000\renders\" /b | find /c ".png"
) else (
    echo ERROR: Renders not found at expected location
    echo Searching for renders...
    dir "%OUTPUT_DIR%\test\" /s /b | findstr "\.png"
)

REM Test backup process
echo.
echo ========================================
echo Testing Backup Process
echo ========================================
set "RENDER_BACKUP_DIR=%RESULTS_DIR%\test_renders\%CONFIG_NAME%"
if exist "%RENDER_BACKUP_DIR%" rmdir /s /q "%RENDER_BACKUP_DIR%"
mkdir "%RENDER_BACKUP_DIR%" 2>nul

if exist "%OUTPUT_DIR%\test\ours_7000\renders" (
    echo Copying renders...
    xcopy "%OUTPUT_DIR%\test\ours_7000\renders\*.*" "%RENDER_BACKUP_DIR%\" /q /y
    if not errorlevel 1 (
        echo SUCCESS: Renders backed up to %RENDER_BACKUP_DIR%
        dir "%RENDER_BACKUP_DIR%" /b | find /c ".png"
    ) else (
        echo ERROR: Backup failed
    )
) else (
    echo ERROR: No renders to backup
)

REM Test metrics calculation
echo.
echo ========================================
echo Testing Metrics Calculation
echo ========================================
python "%GAUSSIAN_SPLATTING_DIR%\metrics.py" -m "%OUTPUT_DIR%" --grayscale > "%RESULTS_DIR%\test_metrics.txt" 2>&1

if not errorlevel 1 (
    echo SUCCESS: Metrics calculated
    echo Results:
    type "%RESULTS_DIR%\test_metrics.txt"
) else (
    echo ERROR: Metrics calculation failed
    type "%RESULTS_DIR%\test_metrics.txt"
)

echo.
echo ========================================
echo Test Complete
echo ========================================
pause