@echo off
REM Spine Analyzer Quick Start Script for Windows
REM This script helps you quickly set up and run the Spine Analyzer system

setlocal enabledelayedexpansion

REM Colors for output (Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

:menu
cls
echo ================================
echo    Spine Analyzer Quick Start
echo ================================
echo 1. Full Setup (Recommended for first time)
echo 2. Start Services
echo 3. Stop Services
echo 4. View Logs
echo 5. Test Installation
echo 6. Process Sample Study
echo 7. Clean Up (Remove all data)
echo 8. Exit
echo ================================
set /p choice="Select an option [1-8]: "

if "%choice%"=="1" goto full_setup
if "%choice%"=="2" goto start_services
if "%choice%"=="3" goto stop_services
if "%choice%"=="4" goto view_logs
if "%choice%"=="5" goto test_installation
if "%choice%"=="6" goto process_study
if "%choice%"=="7" goto cleanup
if "%choice%"=="8" goto exit

echo Invalid option. Please select 1-8.
pause
goto menu

:full_setup
echo %GREEN%[INFO]%NC% Starting full setup...

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERROR]%NC% Docker is not installed. Please install Docker Desktop first.
    pause
    goto menu
)
echo %GREEN%[INFO]%NC% Docker is installed

REM Check Docker Compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERROR]%NC% Docker Compose is not installed. Please install Docker Compose first.
    pause
    goto menu
)
echo %GREEN%[INFO]%NC% Docker Compose is installed

REM Check .env file
if not exist .env (
    echo %YELLOW%[WARNING]%NC% .env file not found. Creating from .env.example...
    if exist .env.example (
        copy .env.example .env
        echo %GREEN%[INFO]%NC% .env file created. Please update it with your configuration.
        notepad .env
    ) else (
        echo %RED%[ERROR]%NC% .env.example file not found. Cannot create .env file.
        pause
        goto menu
    )
) else (
    echo %GREEN%[INFO]%NC% .env file exists
)

REM Check GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%[WARNING]%NC% NVIDIA GPU not detected. Triton may run in CPU mode (slower).
    set /p continue="Continue without GPU? (y/n): "
    if /i not "!continue!"=="y" goto menu
) else (
    echo %GREEN%[INFO]%NC% NVIDIA GPU detected
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)

REM Pull images
echo %GREEN%[INFO]%NC% Pulling Docker images...
docker-compose pull

REM Build images
echo %GREEN%[INFO]%NC% Building custom Docker images...
docker-compose build

REM Start services
echo %GREEN%[INFO]%NC% Starting services...
docker-compose up -d

echo %GREEN%[INFO]%NC% Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check services
echo %GREEN%[INFO]%NC% Checking service status...
docker-compose ps

echo %GREEN%[INFO]%NC% Setup complete!
echo.
echo Access points:
echo   - Orthanc Web UI: http://localhost:8042
echo   - Pipeline API: http://localhost:8000/docs
pause
goto menu

:start_services
echo %GREEN%[INFO]%NC% Starting services...
docker-compose up -d
echo %GREEN%[INFO]%NC% Services started.
pause
goto menu

:stop_services
echo %GREEN%[INFO]%NC% Stopping services...
docker-compose down
echo %GREEN%[INFO]%NC% Services stopped.
pause
goto menu

:view_logs
echo %GREEN%[INFO]%NC% Showing logs (press Ctrl+C to exit)...
docker-compose logs -f
pause
goto menu

:test_installation
echo %GREEN%[INFO]%NC% Testing the installation...

REM Check Orthanc
curl -s -o nul -w "%%{http_code}" http://localhost:8042/system | findstr "200" >nul
if %errorlevel% equ 0 (
    echo %GREEN%[INFO]%NC% Orthanc is running
) else (
    echo %YELLOW%[WARNING]%NC% Orthanc is not responding. Check logs: docker-compose logs orthanc
)

REM Check Pipeline API
curl -s -o nul -w "%%{http_code}" http://localhost:8000/docs | findstr "200" >nul
if %errorlevel% equ 0 (
    echo %GREEN%[INFO]%NC% Pipeline API is running
    echo %GREEN%[INFO]%NC% API documentation: http://localhost:8000/docs
) else (
    echo %YELLOW%[WARNING]%NC% Pipeline API is not responding. Check logs: docker-compose logs pipeline
)

REM Check Triton
curl -s -o nul -w "%%{http_code}" http://localhost:8000/v2/health/ready | findstr "200" >nul
if %errorlevel% equ 0 (
    echo %GREEN%[INFO]%NC% Triton Inference Server is running
) else (
    echo %YELLOW%[WARNING]%NC% Triton is not responding. Check logs: docker-compose logs triton
)

pause
goto menu

:process_study
set /p study_id="Enter Orthanc Study ID: "
echo %GREEN%[INFO]%NC% Processing study %study_id%...
curl -X POST http://localhost:8000/process-study/ -H "Content-Type: application/x-www-form-urlencoded" -d "study_id=%study_id%"
pause
goto menu

:cleanup
echo %YELLOW%[WARNING]%NC% This will remove all containers, volumes, and data!
set /p confirm="Are you sure? (y/n): "
if /i "%confirm%"=="y" (
    docker-compose down -v
    echo %GREEN%[INFO]%NC% Clean up complete.
)
pause
goto menu

:exit
echo %GREEN%[INFO]%NC% Goodbye!
exit /b 0