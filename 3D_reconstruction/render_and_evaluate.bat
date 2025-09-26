@echo off
REM Render and Evaluate Existing 3DGS Models
REM Usage: render_and_evaluate.bat [dataset_name] [method_name]
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
set "WEIGHTS_DIR=%RESULTS_DIR%\weights"

echo ========================================
echo 3DGS Render and Evaluate Existing Models
echo ========================================
echo Dataset: %DATASET_NAME%
echo Method: %METHOD_NAME%
echo Weights Directory: %WEIGHTS_DIR%
echo ========================================

REM ==============================================
REM VALIDATION
REM ==============================================
if not exist "!DATASET_DIR!" (
    echo ERROR: Dataset directory not found: !DATASET_DIR!
    pause
    exit /b 1
)

if not exist "!WEIGHTS_DIR!" (
    echo ERROR: Weights directory not found: !WEIGHTS_DIR!
    echo No trained models available for rendering
    pause
    exit /b 1
)

REM ==============================================
REM FIND AVAILABLE TRAINED MODELS
REM ==============================================
echo Scanning for trained models...
set "AVAILABLE_MODELS="
set "MODEL_COUNT=0"

for /d %%i in ("%WEIGHTS_DIR%\*") do (
    set "MODEL_NAME=%%~nxi"
    if exist "%%i\point_cloud\iteration_7000\point_cloud.ply" (
        set "AVAILABLE_MODELS=!AVAILABLE_MODELS! !MODEL_NAME!"
        set /a MODEL_COUNT+=1
        echo Found trained model: !MODEL_NAME!
    )
)

if %MODEL_COUNT%==0 (
    echo ERROR: No trained models found with iteration_7000
    echo Available directories in weights:
    dir "%WEIGHTS_DIR%" /b
    pause
    exit /b 1
)

echo Found %MODEL_COUNT% trained model(s): %AVAILABLE_MODELS%

REM ==============================================
REM SETUP DIRECTORIES
REM ==============================================
set "RENDER_OUTPUT_DIR=%RESULTS_DIR%\final_renders"
set "METRICS_OUTPUT_DIR=%RESULTS_DIR%\final_metrics"

if not exist "%RENDER_OUTPUT_DIR%" mkdir "%RENDER_OUTPUT_DIR%"
if not exist "%METRICS_OUTPUT_DIR%" mkdir "%METRICS_OUTPUT_DIR%"

echo Output directories:
echo   Renders: %RENDER_OUTPUT_DIR%
echo   Metrics: %METRICS_OUTPUT_DIR%

REM ==============================================
REM PROCESS EACH TRAINED MODEL
REM ==============================================
set "SUCCESS_COUNT=0"
set "FAILED_MODELS="
set "SUCCESS_MODELS="

for %%i in (%AVAILABLE_MODELS%) do (
    set "MODEL_NAME=%%i"
    call :RenderAndEvaluateModel "!MODEL_NAME!"
)

REM ==============================================
REM GENERATE FINAL REPORT
REM ==============================================
echo.
echo ========================================
echo Render and Evaluation Complete
echo ========================================
echo Dataset: %DATASET_NAME%
echo Method: %METHOD_NAME%
echo Total models processed: %MODEL_COUNT%
echo Successful renders: %SUCCESS_COUNT%
set /a FAILED_COUNT=%MODEL_COUNT%-%SUCCESS_COUNT%
echo Failed renders: %FAILED_COUNT%

if not "%FAILED_MODELS%"=="" echo Failed models:%FAILED_MODELS%
if not "%SUCCESS_MODELS%"=="" echo Successful models:%SUCCESS_MODELS%

echo.
echo Results saved to:
echo   Renders: %RENDER_OUTPUT_DIR%
echo   Metrics: %METRICS_OUTPUT_DIR%

REM Generate comparison report
call :GenerateComparisonReport

echo.
echo Press any key to exit...
pause
exit /b 0

REM ==============================================
REM SUBROUTINE: Render and Evaluate Single Model
REM ==============================================
:RenderAndEvaluateModel
set "MODEL_NAME=%~1"
set "SOURCE_WEIGHTS=%WEIGHTS_DIR%\%MODEL_NAME%"
set "TEMP_OUTPUT_DIR=%GAUSSIAN_SPLATTING_DIR%\temp_render_%MODEL_NAME%"

echo.
echo ========================================
echo Processing Model: %MODEL_NAME%
echo ========================================

REM 1. Setup temporary rendering environment
echo Setting up temporary render environment...
if exist "%TEMP_OUTPUT_DIR%" rmdir /s /q "%TEMP_OUTPUT_DIR%"
xcopy "%SOURCE_WEIGHTS%" "%TEMP_OUTPUT_DIR%" /e /i /q >nul

if errorlevel 1 (
    echo ERROR: Failed to copy model weights for %MODEL_NAME%
    set "FAILED_MODELS=!FAILED_MODELS! %MODEL_NAME%"
    exit /b 1
)

REM 2. Render test set (200 images)
echo Rendering test set for %MODEL_NAME%...
python "%GAUSSIAN_SPLATTING_DIR%\render.py" -m "%TEMP_OUTPUT_DIR%" --skip_train --grayscale

if errorlevel 1 (
    echo ERROR: Rendering failed for %MODEL_NAME%
    set "FAILED_MODELS=!FAILED_MODELS! %MODEL_NAME%"
    goto :CleanupModel
)

REM 3. Check and backup rendered images
echo Checking render results...
set "RENDER_SOURCE=%TEMP_OUTPUT_DIR%\test\ours_7000\renders"
set "RENDER_DEST=%RENDER_OUTPUT_DIR%\%MODEL_NAME%"

if exist "%RENDER_SOURCE%" (
    echo Backing up %MODEL_NAME% renders...
    if exist "%RENDER_DEST%" rmdir /s /q "%RENDER_DEST%"
    mkdir "%RENDER_DEST%" 2>nul
    
    xcopy "%RENDER_SOURCE%\*.*" "%RENDER_DEST%\" /q /y >nul
    if not errorlevel 1 (
        echo SUCCESS: %MODEL_NAME% renders backed up
        
        REM Count rendered images
        for /f %%c in ('dir "%RENDER_DEST%\*.png" /b 2^>nul ^| find /c /v ""') do set "IMAGE_COUNT=%%c"
        echo   - Images rendered: !IMAGE_COUNT!
    ) else (
        echo WARNING: Failed to backup renders for %MODEL_NAME%
        set "FAILED_MODELS=!FAILED_MODELS! %MODEL_NAME%"
        goto :CleanupModel
    )
) else (
    echo ERROR: No renders found for %MODEL_NAME% at %RENDER_SOURCE%
    echo Available directories:
    dir "%TEMP_OUTPUT_DIR%\test\" /b 2>nul
    set "FAILED_MODELS=!FAILED_MODELS! %MODEL_NAME%"
    goto :CleanupModel
)

REM 4. Calculate evaluation metrics
echo Calculating metrics for %MODEL_NAME%...
set "METRICS_FILE=%METRICS_OUTPUT_DIR%\%MODEL_NAME%_metrics.txt"

python "%GAUSSIAN_SPLATTING_DIR%\metrics.py" -m "%TEMP_OUTPUT_DIR%" --grayscale > "%METRICS_FILE%" 2>&1

if not errorlevel 1 (
    echo SUCCESS: %MODEL_NAME% metrics calculated
    echo   - Results saved to: %MODEL_NAME%_metrics.txt
) else (
    echo WARNING: Metrics calculation failed for %MODEL_NAME%
    echo Check %METRICS_FILE% for details
)

REM 5. Success
set /a SUCCESS_COUNT+=1
set "SUCCESS_MODELS=!SUCCESS_MODELS! %MODEL_NAME!"
echo SUCCESS: %MODEL_NAME% processing completed

:CleanupModel
REM Cleanup temporary files
echo Cleaning up temporary files for %MODEL_NAME%...
if exist "%TEMP_OUTPUT_DIR%" rmdir /s /q "%TEMP_OUTPUT_DIR%"
exit /b 0

REM ==============================================
REM SUBROUTINE: Generate Comparison Report
REM ==============================================
:GenerateComparisonReport
set "COMPARISON_FILE=%METRICS_OUTPUT_DIR%\comparison_report.txt"

echo Generating comparison report...
echo 3DGS Model Comparison Report - %DATE% %TIME% > "%COMPARISON_FILE%"
echo ============================================== >> "%COMPARISON_FILE%"
echo Dataset: %DATASET_NAME% >> "%COMPARISON_FILE%"
echo Method: %METHOD_NAME% >> "%COMPARISON_FILE%"
echo Models Processed: %SUCCESS_COUNT% >> "%COMPARISON_FILE%"
echo. >> "%COMPARISON_FILE%"

REM Extract metrics from individual files
for %%i in (%SUCCESS_MODELS%) do (
    set "MODEL_NAME=%%i"
    set "METRICS_FILE=%METRICS_OUTPUT_DIR%\!MODEL_NAME!_metrics.txt"
    
    if exist "!METRICS_FILE!" (
        echo Model: !MODEL_NAME! >> "%COMPARISON_FILE%"
        echo ---------------------------------------- >> "%COMPARISON_FILE%"
        
        REM Extract key metrics
        for /f "tokens=2 delims=:" %%j in ('findstr "SSIM" "!METRICS_FILE!" 2^>nul') do (
            echo   SSIM : %%j >> "%COMPARISON_FILE%"
        )
        for /f "tokens=2 delims=:" %%j in ('findstr "PSNR" "!METRICS_FILE!" 2^>nul') do (
            echo   PSNR : %%j >> "%COMPARISON_FILE%"
        )
        for /f "tokens=2 delims=:" %%j in ('findstr "LPIPS" "!METRICS_FILE!" 2^>nul') do (
            echo   LPIPS: %%j >> "%COMPARISON_FILE%"
        )
        echo. >> "%COMPARISON_FILE%"
    )
)

echo SUCCESS: Comparison report saved to comparison_report.txt
exit /b 0