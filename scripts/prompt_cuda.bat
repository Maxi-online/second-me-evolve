@echo off
REM Script to prompt user for CUDA support preference

echo === CUDA Support Selection ===
echo.
echo Do you want to build with NVIDIA GPU (CUDA) support?
echo This requires an NVIDIA GPU and proper NVIDIA Docker runtime configuration.
echo.
set /p choice="Build with CUDA support? (y/n): "

if /i "%choice%"=="y" goto cuda
if /i "%choice%"=="yes" goto cuda
goto nocuda

:cuda
echo Selected: Build WITH CUDA support

REM Create or update .env file with the Dockerfile selection
if exist .env (
    REM Check if variable already exists in file
    findstr /c:"DOCKER_BACKEND_DOCKERFILE" .env >nul
    if %ERRORLEVEL% EQU 0 (
        REM Update existing variable
        powershell -Command "(Get-Content .env) -replace '^DOCKER_BACKEND_DOCKERFILE=.*', 'DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda' | Set-Content .env"
    ) else (
        REM Append to file with newline before
        echo.>> .env
        echo DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda>> .env
    )
) else (
    REM Create new file
    echo DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda> .env
)

REM Create a flag file to indicate GPU use
echo GPU > .gpu_selected

echo Environment set to build with CUDA support
goto end

:nocuda
echo Selected: Build WITHOUT CUDA support (CPU only)

REM Create or update .env file with the Dockerfile selection
if exist .env (
    REM Check if variable already exists in file
    findstr /c:"DOCKER_BACKEND_DOCKERFILE" .env >nul
    if %ERRORLEVEL% EQU 0 (
        REM Update existing variable
        powershell -Command "(Get-Content .env) -replace '^DOCKER_BACKEND_DOCKERFILE=.*', 'DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend' | Set-Content .env"
    ) else (
        REM Append to file with newline before
        echo.>> .env
        echo DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend>> .env
    )
) else (
    REM Create new file
    echo DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend> .env
)

REM Remove any GPU flag file if it exists
if exist .gpu_selected (
    del .gpu_selected
)

echo Environment set to build without CUDA support

:end
echo === CUDA Selection Complete ===