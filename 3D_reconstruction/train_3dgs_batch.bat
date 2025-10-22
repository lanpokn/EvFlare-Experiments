@echo off
REM 3DGS Batch Training Script - Training Only
REM Usage: train_3dgs_batch.bat [dataset_name] [method_name]
REM Purpose: Train multiple 3DGS configurations, save weights for later rendering
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
echo 3DGS Batch Training (Training Only)
echo ========================================
echo Dataset: %DATASET_NAME%
echo Method: %METHOD_NAME%
echo Note: Use render_and_evaluate.py for rendering after training
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

REM ==============================================
REM BACKUP ORIGINAL CONFIG
REM ==============================================
if exist "!DATASET_DIR!\transforms_train.json" (
    copy "!DATASET_DIR!\transforms_train.json" "!DATASET_DIR!\transforms_train_backup.json" >nul
    echo Backed up original config: transforms_train_backup.json
)

REM ==============================================
REM PROCESS EACH CONFIGURATION - TRAINING ONLY
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
    call :TrainSingleConfig "!CONFIG_NAME!" !CURRENT_COUNT! !TOTAL_COUNT!
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
echo Training Complete
echo ========================================
echo Dataset: %DATASET_NAME%
echo Method: %METHOD_NAME%
echo Total configurations: %TOTAL_COUNT%
echo Successful trainings: %SUCCESS_COUNT%
set /a FAILED_COUNT=%TOTAL_COUNT%-%SUCCESS_COUNT%
echo Failed trainings: %FAILED_COUNT%

if not "%FAILED_METHODS%"=="" echo Failed configs:%FAILED_METHODS%
if not "%SUCCESS_METHODS%"=="" echo Successful configs:%SUCCESS_METHODS%

echo.
echo Trained weights saved to: %RESULTS_DIR%\weights
echo.
echo NEXT STEP: Run rendering and evaluation:
echo   python render_and_evaluate.py --dataset %DATASET_NAME% --method %METHOD_NAME% --weights-dir "gaussian-splatting/output"
pause
exit /b 0

REM ==============================================
REM SUBROUTINE: Train Single Configuration
REM ==============================================
:TrainSingleConfig
set "CONFIG_NAME=%~1"
set "CURRENT=%~2"
set "TOTAL=%~3"

echo.
echo ========================================
echo Training: %CONFIG_NAME% ^(Progress: %CURRENT%/%TOTAL%^)
echo ========================================

REM Check if config file exists
set "CONFIG_FILE=!DATASET_DIR!\transforms_train_%CONFIG_NAME%.json"
if not exist "%CONFIG_FILE%" (
    echo ERROR: Config file not found: %CONFIG_FILE%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

echo [1/2] Switching config: %CONFIG_NAME%
copy "%CONFIG_FILE%" "!DATASET_DIR!\transforms_train.json" >nul
if errorlevel 1 (
    echo ERROR: Config file switch failed: %CONFIG_NAME%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

REM ==============================================
REM TRAINING WITH --eval
REM ==============================================
echo [2/2] Training with --eval mode: %CONFIG_NAME%
set "OUTPUT_DIR=%GAUSSIAN_SPLATTING_DIR%\output\%DATASET_NAME%_%CONFIG_NAME%"

REM CRITICAL: Add --eval for proper train/test separation
python "%GAUSSIAN_SPLATTING_DIR%\train.py" -s "!DATASET_DIR!" -m "%OUTPUT_DIR%" --iterations 10000 --grayscale --eval

if errorlevel 1 (
    echo ERROR: Training failed: %CONFIG_NAME%
    set "FAILED_METHODS=!FAILED_METHODS! %CONFIG_NAME%"
    exit /b 1
)

echo SUCCESS: Training completed: %CONFIG_NAME%

REM ==============================================
REM BACKUP WEIGHTS ONLY
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

echo SUCCESS: Weights backed up: %CONFIG_NAME%

set /a SUCCESS_COUNT+=1
set "SUCCESS_METHODS=!SUCCESS_METHODS! %CONFIG_NAME%"
echo SUCCESS: Configuration %CONFIG_NAME% training completed
exit /b 0