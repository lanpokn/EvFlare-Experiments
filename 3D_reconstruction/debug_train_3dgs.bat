@echo off
REM Debug version to find the issue
setlocal EnableDelayedExpansion

REM Set parameters
set "DATASET_NAME=%1"
if "%DATASET_NAME%"=="" set "DATASET_NAME=lego2"

set "METHOD_NAME=%2"
if "%METHOD_NAME%"=="" set "METHOD_NAME=spade_e2vid"

set "DATASET_DIR=datasets\%DATASET_NAME%"
set "METHODS_FILE=!DATASET_DIR!\training_methods_!METHOD_NAME!.txt"

echo DEBUG: DATASET_DIR = !DATASET_DIR!
echo DEBUG: METHODS_FILE = !METHODS_FILE!

if not exist "!METHODS_FILE!" (
    echo ERROR: Method config file not found: !METHODS_FILE!
    pause
    exit /b 1
)

echo DEBUG: Method file exists, reading contents...
type "!METHODS_FILE!"

echo.
echo DEBUG: Processing each configuration...

REM Process each training configuration
set "CURRENT_COUNT=0"
for /f "tokens=*" %%i in (!METHODS_FILE!) do (
    set "CONFIG_NAME=%%i"
    set /a CURRENT_COUNT+=1
    
    echo.
    echo DEBUG: Processing line %%i
    echo DEBUG: CONFIG_NAME = !CONFIG_NAME!
    echo DEBUG: CURRENT_COUNT = !CURRENT_COUNT!
    echo DEBUG: Looking for: !DATASET_DIR!\transforms_train_!CONFIG_NAME!.json
    
    if exist "!DATASET_DIR!\transforms_train_!CONFIG_NAME!.json" (
        echo DEBUG: Config file EXISTS for !CONFIG_NAME!
    ) else (
        echo DEBUG: Config file NOT FOUND for !CONFIG_NAME!
        echo DEBUG: Available files:
        dir "!DATASET_DIR!\transforms_train_*.json" /b 2>nul
    )
    
    echo DEBUG: End of processing for !CONFIG_NAME!
)

echo.
echo DEBUG: Finished processing all configurations
pause