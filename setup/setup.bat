@echo off
setlocal

rem Ensure all venvs are created in the repository root, not inside the setup folder.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."
set "ROOT_DIR=%CD%"

echo Checking Python version...

set PYVER=
set PYTHON_EXE=

for /f "delims=" %%P in ('where python 2^>nul') do (
    call :check_python_candidate "%%~P"
    if defined PYTHON_EXE goto :FoundPython
)

if not defined PYTHON_EXE (
    echo Current python is not 3.10.11 or no python on PATH. Checking py launcher for Python 3.10...
    set PYVER=
    for /f "delims=" %%i in ('py -3.10 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2^>nul') do set PYVER=%%i
    if "%PYVER%"=="3.10.11" (
        set "PYTHON_EXE=py -3.10"
        echo Found global Python 3.10.11 via py launcher.
    )
)

:FoundPython
if not "%PYVER%"=="3.10.11" (
    echo ERROR: Invalid Python version detected: %PYVER%
    echo Required version: 3.10.11
    if exist python-3.10.11-amd64.exe (
        echo Launching local installer: python-3.10.11-amd64.exe
        start /wait "" python-3.10.11-amd64.exe
        echo After installation finishes, rerun setup.bat.
    ) else (
        echo Python 3.10.11 is not installed. Install it manually and rerun setup.bat.
    )
    pause
    exit /b 1
)

echo Python version OK: %PYVER% (using %PYTHON_EXE%)

goto :Continue

:check_python_candidate
setlocal
set "CAND=%~1"
if defined VIRTUAL_ENV (
    if /i "%CAND%"=="%VIRTUAL_ENV%\Scripts\python.exe" endlocal & goto :EOF
)
echo %CAND% | findstr /I /C:"\Scripts\python.exe" >nul
if not errorlevel 1 endlocal & goto :EOF
set "CHECKVER="
for /f "delims=" %%V in ('"%CAND%" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2^>nul') do set "CHECKVER=%%V"
if "%CHECKVER%"=="3.10.11" (
    endlocal & set "PYTHON_EXE=%CAND%" & set "PYVER=%CHECKVER%"
) else (
    endlocal
)
goto :EOF

:Continue

echo ==========================
echo Setting up main venv
echo ==========================

set "VENV_MAIN=%ROOT_DIR%\venv"
call :ensure_venv "%VENV_MAIN%" "%ROOT_DIR%\requirements.txt"

echo ==========================
echo Setting up DeepFace venv
echo ==========================

set "VENV_DEEPFACE=%ROOT_DIR%\venv-deepface"

if exist "%VENV_DEEPFACE%\Scripts\activate.bat" (
    echo DeepFace venv already exists, skipping creation and dependency install.
) else (
    set /p CREATE_DEEPFACE="DeepFace venv not found. Create venv-deepface and install dependencies? [y/N]: "
    if /i "%CREATE_DEEPFACE%"=="y" (
        call :ensure_venv "%VENV_DEEPFACE%" "%ROOT_DIR%\requirements-deepface.txt"
    ) else (
        echo Skipping DeepFace venv creation as requested.
    )
)

goto :Done

:ensure_venv
setlocal
set "VENV_PATH=%~1"
set "REQ_FILE=%~2"
if exist "%VENV_PATH%\Scripts\activate.bat" (
    endlocal
    goto :EOF
)
echo Creating venv at %VENV_PATH%...
%PYTHON_EXE% -m venv "%VENV_PATH%"

if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo ERROR: Failed to create venv at %VENV_PATH%
    pause
    popd
    exit /b 1
)

call "%VENV_PATH%\Scripts\activate.bat"
python -m pip install --upgrade pip

if exist "%REQ_FILE%" (
    pip install -r "%REQ_FILE%"
) else (
    echo No %REQ_FILE% found
)

deactivate
endlocal

goto :EOF

:Done

echo ==========================
echo Setup complete
echo ==========================

popd
pause