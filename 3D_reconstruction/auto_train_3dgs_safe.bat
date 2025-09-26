@echo off
REM Safe 3DGS Training Script - No GOTO, Simple Logic
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
echo 3D Gaussian Splatting Auto Training
echo ========================================
echo Dataset: %DATASET_NAME%
echo Method: %METHOD_NAME%
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
if not exist "%RESULTS_DIR%\renders" mkdir "%RESULTS_DIR%\renders"
if not exist "%RESULTS_DIR%\metrics" mkdir "%RESULTS_DIR%\metrics"

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
echo Will train !TOTAL_COUNT! configurations

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
REM GENERATE FINAL REPORT
REM ==============================================
echo.
echo ========================================
echo Batch Processing Complete Report
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
echo Results saved to: %RESULTS_DIR%
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

echo Switching config: %CONFIG_NAME%
copy "%CONFIG_FILE%" "!DATASET_DIR!\transforms_train.json" >nul
if errorlevel 1 (
    echo ERROR: Config file switch failed: %CONFIG_NAME%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

REM Train 3DGS model
echo Training: %CONFIG_NAME%
set "OUTPUT_DIR=%GAUSSIAN_SPLATTING_DIR%\output\%DATASET_NAME%_%CONFIG_NAME%"
python "%GAUSSIAN_SPLATTING_DIR%\train.py" -s "!DATASET_DIR!" -m "%OUTPUT_DIR%" --iterations 7000 --grayscale

if errorlevel 1 (
    echo ERROR: Training failed: %CONFIG_NAME%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

echo SUCCESS: Training completed: %CONFIG_NAME%

REM Backup weights
echo Backing up weights: %CONFIG_NAME%
set "WEIGHT_BACKUP_DIR=%RESULTS_DIR%\weights\%CONFIG_NAME%"
if exist "%WEIGHT_BACKUP_DIR%" rmdir /s /q "%WEIGHT_BACKUP_DIR%"
xcopy "%OUTPUT_DIR%" "%WEIGHT_BACKUP_DIR%" /e /i /q >nul
if not errorlevel 1 echo SUCCESS: Weights backed up

REM Training completed - weights will be backed up
echo Training phase completed for %CONFIG_NAME%
echo Use render_and_evaluate.bat to generate renders and calculate metrics

REM Clean up temporary output directory AFTER backing up everything
echo Cleaning up temporary files: %CONFIG_NAME%
if exist "%OUTPUT_DIR%" (
    rmdir /s /q "%OUTPUT_DIR%"
    echo DEBUG: Cleaned up %OUTPUT_DIR%
)

set /a SUCCESS_COUNT+=1
set "SUCCESS_METHODS=!SUCCESS_METHODS! %CONFIG_NAME%"
echo SUCCESS: Configuration %CONFIG_NAME% completed
exit /b 0