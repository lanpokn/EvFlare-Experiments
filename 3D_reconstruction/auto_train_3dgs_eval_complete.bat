@echo off
REM Complete 3DGS Training & Evaluation Script with --eval support
REM Fixed: Train/Test separation + Render + Metrics calculation
setlocal EnableDelayedExpansion

REM ==============================================
REM INITIALIZATION
REM ==============================================
set "DATASET_NAME=%1"
if "%DATASET_NAME%"=="" set "DATASET_NAME=lego2"

set "METHOD_NAME=%2"
if "%METHOD_NAME%"=="" set "METHOD_NAME=spade_e2vid"

set "DATASET_DIR=datasets\%DATASET_NAME%"
set "RESULTS_DIR=%DATASET_DIR%\3dgs_results"
set "GAUSSIAN_SPLATTING_DIR=gaussian-splatting"
set "METHODS_FILE=!DATASET_DIR!\training_methods_!METHOD_NAME!.txt"

echo ========================================
echo Complete 3DGS Training & Evaluation
echo ========================================
echo Dataset: %DATASET_NAME%
echo Method: %METHOD_NAME%
echo NEW: --eval mode for proper train/test separation
echo NEW: Automatic render generation after training
echo NEW: Automatic metrics calculation
echo ========================================

REM ==============================================
REM VALIDATION
REM ==============================================
if not exist "!DATASET_DIR!" (
    echo ERROR: Dataset directory not found: !DATASET_DIR!
    pause
    exit /b 1
)

if not exist "!GAUSSIAN_SPLATTING_DIR!" (
    echo ERROR: 3DGS directory not found: !GAUSSIAN_SPLATTING_DIR!
    pause
    exit /b 1
)

if not exist "!METHODS_FILE!" (
    echo ERROR: Method config file not found: !METHODS_FILE!
    echo Please run: python generate_json_configs.py !DATASET_NAME! !METHOD_NAME!
    pause
    exit /b 1
)

REM ==============================================
REM SETUP DIRECTORIES
REM ==============================================
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%RESULTS_DIR%\weights" mkdir "%RESULTS_DIR%\weights"
if not exist "%RESULTS_DIR%\final_renders" mkdir "%RESULTS_DIR%\final_renders"
if not exist "%RESULTS_DIR%\final_metrics" mkdir "%RESULTS_DIR%\final_metrics"

REM ==============================================
REM BACKUP ORIGINAL CONFIG
REM ==============================================
if exist "!DATASET_DIR!\transforms_train.json" (
    copy "!DATASET_DIR!\transforms_train.json" "!DATASET_DIR!\transforms_train_backup.json" >nul
    echo Backed up original config: transforms_train_backup.json
)

REM ==============================================
REM PROCESS EACH CONFIGURATION
REM ==============================================
set "SUCCESS_COUNT=0"
set "TOTAL_COUNT=0"
set "FAILED_METHODS="
set "SUCCESS_METHODS="

REM Count total configurations
for /f "tokens=*" %%i in (!METHODS_FILE!) do set /a TOTAL_COUNT+=1
echo Will train and evaluate !TOTAL_COUNT! configurations

set "CURRENT_COUNT=0"
for /f "tokens=*" %%i in (!METHODS_FILE!) do (
    set "CONFIG_NAME=%%i"
    set /a CURRENT_COUNT+=1
    call :ProcessConfig "!CONFIG_NAME!" !CURRENT_COUNT! !TOTAL_COUNT!
)

REM ==============================================
REM RESTORE ORIGINAL CONFIG
REM ==============================================
if exist "!DATASET_DIR!\transforms_train_backup.json" (
    copy "!DATASET_DIR!\transforms_train_backup.json" "!DATASET_DIR!\transforms_train.json" >nul
    del "!DATASET_DIR!\transforms_train_backup.json" >nul
    echo Restored original configuration file
)

REM ==============================================
REM FINAL METRICS COMPARISON
REM ==============================================
if !SUCCESS_COUNT! GTR 1 (
    echo.
    echo ========================================
    echo Generating Comparison Report
    echo ========================================
    call :GenerateComparisonReport
)

REM ==============================================
REM GENERATE FINAL REPORT
REM ==============================================
echo.
echo ========================================
echo Complete Processing Report
echo ========================================
echo Dataset: %DATASET_NAME%
echo Method: %METHOD_NAME%
echo Total configurations: %TOTAL_COUNT%
echo Successful configurations: %SUCCESS_COUNT%
set /a FAILED_COUNT=%TOTAL_COUNT%-%SUCCESS_COUNT%
echo Failed configurations: %FAILED_COUNT%

if not "%FAILED_METHODS%"=="" echo Failed configs:%FAILED_METHODS%
if not "%SUCCESS_METHODS%"=="" echo Successful configs:%SUCCESS_METHODS%

echo.
echo Results saved to:
echo   Weights: %RESULTS_DIR%\weights
echo   Renders: %RESULTS_DIR%\final_renders
echo   Metrics: %RESULTS_DIR%\final_metrics
pause
exit /b 0

REM ==============================================
REM SUBROUTINE: Process Single Configuration
REM ==============================================
:ProcessConfig
set "CONFIG_NAME=%~1"
set "CURRENT=%~2"
set "TOTAL=%~3"

echo.
echo ========================================
echo Processing: %CONFIG_NAME% ^(Progress: %CURRENT%/%TOTAL%^)
echo ========================================

REM Check if config file exists
set "CONFIG_FILE=!DATASET_DIR!\transforms_train_%CONFIG_NAME%.json"
if not exist "%CONFIG_FILE%" (
    echo ERROR: Config file not found: %CONFIG_FILE%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

echo [1/3] Switching config: %CONFIG_NAME%
copy "%CONFIG_FILE%" "!DATASET_DIR!\transforms_train.json" >nul
if errorlevel 1 (
    echo ERROR: Config file switch failed: %CONFIG_NAME%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

REM ==============================================
REM PHASE 1: TRAINING WITH --eval
REM ==============================================
echo [2/3] Training with --eval mode: %CONFIG_NAME%
set "OUTPUT_DIR=%GAUSSIAN_SPLATTING_DIR%\output\%DATASET_NAME%_%CONFIG_NAME%"

REM CRITICAL FIX: Add --eval for proper train/test separation
python "%GAUSSIAN_SPLATTING_DIR%\train.py" -s "!DATASET_DIR!" -m "%OUTPUT_DIR%" --iterations 7000 --grayscale --eval

if errorlevel 1 (
    echo ERROR: Training failed: %CONFIG_NAME%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

echo SUCCESS: Training completed: %CONFIG_NAME%

REM ==============================================
REM PHASE 2: RENDERING TEST SET
REM ==============================================
echo [3/3] Rendering test set: %CONFIG_NAME%

REM Change to 3DGS directory for proper execution
pushd "%GAUSSIAN_SPLATTING_DIR%"

python render.py --iteration 7000 -m "%OUTPUT_DIR%" -s "..\..\!DATASET_DIR!" --skip_train --grayscale

if errorlevel 1 (
    echo ERROR: Rendering failed: %CONFIG_NAME%
    popd
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

popd
echo SUCCESS: Rendering completed: %CONFIG_NAME%

REM ==============================================
REM PHASE 3: BACKUP WEIGHTS AND RENDERS
REM ==============================================
echo Backing up weights: %CONFIG_NAME%
set "WEIGHT_BACKUP_DIR=%RESULTS_DIR%\weights\%CONFIG_NAME%"
if exist "%WEIGHT_BACKUP_DIR%" rmdir /s /q "%WEIGHT_BACKUP_DIR%"
xcopy "%OUTPUT_DIR%" "%WEIGHT_BACKUP_DIR%" /e /i /q >nul
if errorlevel 1 (
    echo ERROR: Weight backup failed: %CONFIG_NAME%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

REM Backup renders to final location
echo Backing up renders: %CONFIG_NAME%
set "RENDER_SOURCE_DIR=%OUTPUT_DIR%\test\ours_7000\renders"
set "RENDER_DEST_DIR=%RESULTS_DIR%\final_renders\%CONFIG_NAME%"

if exist "%RENDER_SOURCE_DIR%" (
    if exist "%RENDER_DEST_DIR%" rmdir /s /q "%RENDER_DEST_DIR%"
    xcopy "%RENDER_SOURCE_DIR%" "%RENDER_DEST_DIR%" /e /i /q >nul
    
    REM Count rendered images
    set "RENDER_COUNT=0"
    for %%f in ("%RENDER_DEST_DIR%\*.png") do set /a RENDER_COUNT+=1
    echo SUCCESS: !RENDER_COUNT! renders backed up
) else (
    echo WARNING: No renders found at %RENDER_SOURCE_DIR%
)

REM ==============================================
REM PHASE 4: CALCULATE METRICS
REM ==============================================
echo Calculating metrics: %CONFIG_NAME%

REM Change to 3DGS directory for metrics calculation
pushd "%GAUSSIAN_SPLATTING_DIR%"

python metrics.py -m "%OUTPUT_DIR%" --grayscale > "..\%RESULTS_DIR%\final_metrics\%CONFIG_NAME%_metrics.txt" 2>&1

if errorlevel 1 (
    echo WARNING: Metrics calculation failed: %CONFIG_NAME%
) else (
    echo SUCCESS: Metrics calculated: %CONFIG_NAME%
)

popd

REM Clean up temporary output directory AFTER backing up everything
echo Cleaning up temporary files: %CONFIG_NAME%
if exist "%OUTPUT_DIR%" (
    rmdir /s /q "%OUTPUT_DIR%"
    echo Cleaned up temporary directory
)

set /a SUCCESS_COUNT+=1
set "SUCCESS_METHODS=!SUCCESS_METHODS! %CONFIG_NAME%"
echo SUCCESS: Configuration %CONFIG_NAME% completed (Train + Render + Metrics)
exit /b 0

REM ==============================================
REM SUBROUTINE: Generate Comparison Report
REM ==============================================
:GenerateComparisonReport
echo Generating comparison report...
set "REPORT_FILE=%RESULTS_DIR%\final_metrics\comparison_report.txt"

echo 3DGS Model Comparison Report - %date% %time% > "%REPORT_FILE%"
echo ========================================== >> "%REPORT_FILE%"
echo Dataset: %DATASET_NAME% >> "%REPORT_FILE%"
echo Method: %METHOD_NAME% >> "%REPORT_FILE%"
echo Configurations Processed: %SUCCESS_COUNT% >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"

REM Append individual metrics
for %%f in ("%RESULTS_DIR%\final_metrics\*_metrics.txt") do (
    set "CONFIG_FILE=%%~nf"
    set "CONFIG_FILE=!CONFIG_FILE:_metrics=!"
    echo Configuration: !CONFIG_FILE! >> "%REPORT_FILE%"
    echo ---------------------------------------- >> "%REPORT_FILE%"
    type "%%f" >> "%REPORT_FILE%" 2>nul
    echo. >> "%REPORT_FILE%"
)

echo Comparison report saved: %REPORT_FILE%
exit /b 0