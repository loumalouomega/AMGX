@echo off
REM ============================================================================
REM AMGX Build Configuration Script for Windows
REM ============================================================================
REM This script configures the AMGX build using CMake.
REM Run this script from the build directory.
REM
REM Usage: configure.bat [options]
REM   Options:
REM     --release       Build Release configuration (default)
REM     --debug         Build Debug configuration
REM     --profile       Build Profile configuration
REM     --no-mpi        Disable MPI support
REM     --arch <list>   CUDA architectures (e.g., "86;89;90;100;120")
REM     --generator <g> CMake generator (default: "Ninja")
REM     --help          Show this help message
REM
REM Example:
REM     configure.bat --release --arch "90;100"
REM ============================================================================

setlocal enabledelayedexpansion

REM Default configuration values
set BUILD_TYPE=Release
set CMAKE_NO_MPI=OFF
set CUDA_ARCHITECTURES=86;89;90;100;120
set CMAKE_GENERATOR=Ninja
@REM set CMAKE_GENERATOR=Visual Studio 17 2022
set CMAKE_INSTALL_PREFIX=
set SOURCE_DIR=%~dp0..

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--release" (
    set BUILD_TYPE=Release
    shift
    goto :parse_args
)
if /i "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto :parse_args
)
if /i "%~1"=="--profile" (
    set BUILD_TYPE=Profile
    shift
    goto :parse_args
)
if /i "%~1"=="--no-mpi" (
    set CMAKE_NO_MPI=ON
    shift
    goto :parse_args
)
if /i "%~1"=="--arch" (
    set CUDA_ARCHITECTURES=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--generator" (
    set CMAKE_GENERATOR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--install-prefix" (
    set CMAKE_INSTALL_PREFIX=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    goto :show_help
)
echo Warning: Unknown option "%~1"
shift
goto :parse_args

:done_parsing

REM Display configuration
echo ============================================================================
echo AMGX Build Configuration
echo ============================================================================
echo Source Directory:    %SOURCE_DIR%
echo Build Type:          %BUILD_TYPE%
echo CUDA Architectures:  %CUDA_ARCHITECTURES%
echo MPI Disabled:        %CMAKE_NO_MPI%
echo CMake Generator:     %CMAKE_GENERATOR%
echo ============================================================================
echo.

REM Check for CMake
where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake not found in PATH. Please install CMake and add it to PATH.
    exit /b 1
)

REM Check for CUDA
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: CUDA nvcc not found in PATH. Make sure CUDA Toolkit is installed.
    echo          The build may fail if CUDA is not properly configured.
)

REM Run CMake configuration
echo Running CMake configuration...
echo.

set vs2022_path=C:\Program Files (x86)\Microsoft Visual Studio\2022.17.12.4\Common7\Tools\VsDevCmd.bat
call "%vs2022_path%" -arch=amd64

set CMAKE_ARGS=-G "%CMAKE_GENERATOR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_CUDA_ARCHITECTURES="%CUDA_ARCHITECTURES%" -DCMAKE_NO_MPI=%CMAKE_NO_MPI%
if not "%CMAKE_INSTALL_PREFIX%"=="" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_INSTALL_PREFIX="%CMAKE_INSTALL_PREFIX%"
)

cmake %CMAKE_ARGS% "%SOURCE_DIR%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: CMake configuration failed.
    exit /b 1
)

echo.
echo ============================================================================
echo Configuration complete, starting build...
echo ============================================================================

cmake --build . --config %BUILD_TYPE% --target install --parallel 16

exit /b 0

:show_help
echo ============================================================================
echo AMGX Build Configuration Script for Windows
echo ============================================================================
echo.
echo Usage: configure.bat [options]
echo.
echo Options:
echo   --release       Build Release configuration (default)
echo   --debug         Build Debug configuration
echo   --profile       Build Profile configuration
echo   --no-mpi        Disable MPI support (single GPU build)
echo   --arch ^<list^>   CUDA architectures to target
echo                   Default: "86;89;90;100;120"
echo                   Example: --arch "80;90;100"
echo   --generator ^<g^> CMake generator to use
echo                   Default: "Ninja"
echo                   Example: --generator "Visual Studio 17 2022"
echo   --install-prefix ^<path^>
echo                   Installation prefix directory
echo                   Example: --install-prefix "C:\Program Files\AMGX"
echo   --help          Show this help message
echo.
echo Examples:
echo   configure.bat
echo       Configure with default settings (Release, with MPI, Ninja)
echo.
echo   configure.bat --debug --no-mpi
echo       Configure Debug build without MPI support
echo.
echo   configure.bat --release --arch "80;90"
echo       Configure Release build for Ampere and Hopper GPUs
echo.
echo   configure.bat --generator "Visual Studio 17 2022" --release
echo       Configure using Visual Studio build system
echo.
exit /b 0
