#!/bin/bash
# ============================================================================
# AMGX Build Configuration Script for GNU/Linux
# ============================================================================
# This script configures the AMGX build using CMake.
# Run this script from the build directory.
#
# Usage: configure.sh [options]
#   Options:
#     --release       Build Release configuration (default)
#     --debug         Build Debug configuration
#     --profile       Build Profile configuration
#     --no-mpi        Disable MPI support
#     --arch <list>   CUDA architectures (e.g., "86;89;90;100;120")
#     --generator <g> CMake generator (default: "Ninja")
#     --help          Show this help message
#
# Example:
#     ./configure.sh --release --arch "90;100"
# ============================================================================

# Function to show help
show_help() {
    cat << EOF
============================================================================
AMGX Build Configuration Script for GNU/Linux
============================================================================

Usage: configure.sh [options]

Options:
  --release       Build Release configuration (default)
  --debug         Build Debug configuration
  --profile       Build Profile configuration
  --no-mpi        Disable MPI support (single GPU build)
  --arch <list>   CUDA architectures to target
                  Default: "86;89;90;100;120"
                  Example: --arch "80;90;100"
  --generator <g> CMake generator to use
                  Default: "Ninja"
                  Example: --generator "Unix Makefiles"
  --install-prefix <path>
                  Installation prefix directory
                  Example: --install-prefix "/usr/local/amgx"
  --help          Show this help message

Examples:
  ./configure.sh
      Configure with default settings (Release, with MPI, Ninja)

  ./configure.sh --debug --no-mpi
      Configure Debug build without MPI support

  ./configure.sh --release --arch "80;90"
      Configure Release build for Ampere and Hopper GPUs

  ./configure.sh --generator "Unix Makefiles" --release
      Configure using Unix Makefiles build system

EOF
    exit 0
}

# Default configuration values
BUILD_TYPE="Release"
CMAKE_NO_MPI="OFF"
CUDA_ARCHITECTURES="86;89;90;100;120"
CMAKE_GENERATOR="Ninja"
CMAKE_INSTALL_PREFIX=""
SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --profile)
            BUILD_TYPE="Profile"
            shift
            ;;
        --no-mpi)
            CMAKE_NO_MPI="ON"
            shift
            ;;
        --arch)
            CUDA_ARCHITECTURES="$2"
            shift 2
            ;;
        --generator)
            CMAKE_GENERATOR="$2"
            shift 2
            ;;
        --install-prefix)
            CMAKE_INSTALL_PREFIX="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Warning: Unknown option \"$1\""
            shift
            ;;
    esac
done

# Display configuration
cat << EOF
============================================================================
AMGX Build Configuration
============================================================================
Source Directory:    $SOURCE_DIR
Build Type:          $BUILD_TYPE
CUDA Architectures:  $CUDA_ARCHITECTURES
MPI Disabled:        $CMAKE_NO_MPI
CMake Generator:     $CMAKE_GENERATOR
============================================================================

EOF

# Check for CUDA and try to locate it in common installation paths
if ! command -v nvcc > /dev/null 2>&1; then
    echo "WARNING: CUDA nvcc not found in PATH."
    echo "         Searching common CUDA installation locations..."
    
    CUDA_FOUND=false
    for cuda_path in "/usr/local/cuda/bin" "/usr/local/cuda-13/bin" "/usr/local/cuda-12/bin" "/opt/cuda/bin" "$HOME/cuda/bin"; do
        if [ -f "$cuda_path/nvcc" ]; then
            echo "         Found CUDA at: $cuda_path"
            export PATH="$cuda_path:$PATH"
            CUDA_FOUND=true
            break
        fi
    done
    
    if [ "$CUDA_FOUND" = "false" ]; then
        echo "         CUDA not found in common locations."
        echo "         Please add CUDA bin directory to PATH, e.g.:"
        echo "         export PATH=/usr/local/cuda/bin:\$PATH"
        echo "         The build may fail if CUDA is not properly configured."
    fi
fi

# Run CMake configuration
echo "Running CMake configuration..."
echo

cmake -G "$CMAKE_GENERATOR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" -DCMAKE_NO_MPI="$CMAKE_NO_MPI" ${CMAKE_INSTALL_PREFIX:+-DCMAKE_INSTALL_PREFIX="$CMAKE_INSTALL_PREFIX"} "$SOURCE_DIR"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: CMake configuration failed."
    exit 1
fi

echo
cat << EOF
============================================================================
Configuration complete, starting build...
============================================================================
EOF

cmake --build . --config "$BUILD_TYPE" --target install --parallel 16

exit 0
