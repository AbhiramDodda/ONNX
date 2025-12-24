#!/bin/bash

# MiniONNXRuntime Build Script
# Usage: ./build.sh [clean|test|install|help]

set -e  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_info "Checking dependencies..."
    if ! command -v cmake &> /dev/null; then
        print_error "CMake not found. Please install CMake 3.15 or higher."
        exit 1
    fi

    if ! command -v conan &> /dev/null; then
        print_warn "Conan not found. Attempting to install via pip..."
        pip install conan || {
            print_error "Failed to install Conan. Please install manually."
            exit 1
        }
    fi
    
    print_info "All dependencies found."
}

setup_conan() {
    print_info "Setting up Conan..."
    
    # Detect profile if not exists
    if [ ! -f ~/.conan2/profiles/default ]; then
        print_info "Creating default Conan profile..."
        conan profile detect --force
    fi
}

install_dependencies() {
    print_info "Installing dependencies with Conan..."
    
    conan install . \
        --output-folder="${BUILD_DIR}" \
        --build=missing \
        -s build_type="${BUILD_TYPE}"
}

configure() {
    print_info "Configuring CMake (${BUILD_TYPE})..."
    
    cmake -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${BUILD_DIR}/conan_toolchain.cmake" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DBUILD_TESTS=ON \
        -DBUILD_EXAMPLES=ON
}

build() {
    print_info "Building project..."
    
    cmake --build "${BUILD_DIR}" \
        --config "${BUILD_TYPE}" \
        -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
}

run_tests() {
    print_info "Running tests..."
    
    cd "${BUILD_DIR}"
    ctest --output-on-failure -C "${BUILD_TYPE}"
    cd "${SCRIPT_DIR}"
}

run_example() {
    print_info "Running example..."
    
    if [ -f "${BUILD_DIR}/mini_onnx_main" ]; then
        "${BUILD_DIR}/mini_onnx_main" examples/toy_model.json
    elif [ -f "${BUILD_DIR}/${BUILD_TYPE}/mini_onnx_main" ]; then
        "${BUILD_DIR}/${BUILD_TYPE}/mini_onnx_main" examples/toy_model.json
    elif [ -f "${BUILD_DIR}/mini_onnx_main.exe" ]; then
        "${BUILD_DIR}/mini_onnx_main.exe" examples/toy_model.json
    else
        print_error "Executable not found"
        exit 1
    fi
}

clean() {
    print_info "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    print_info "Clean complete."
}

install_project() {
    print_info "Installing project..."
    
    cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX:-/usr/local}"
}

show_help() {
    cat << EOF
MiniONNXRuntime Build Script

Usage: $0 [command]

Commands:
    (none)      - Full build (deps + configure + build)
    clean       - Remove build directory
    test        - Run tests
    example     - Run example application
    install     - Install to system
    help        - Show this help

Environment variables:
    BUILD_TYPE      - Build type (Release|Debug) [default: Release]
    INSTALL_PREFIX  - Install prefix [default: /usr/local]

Examples:
    $0                          # Build project
    BUILD_TYPE=Debug $0         # Build in debug mode
    $0 clean                    # Clean build directory
    $0 test                     # Run tests

EOF
}

main() {
    local command="${1:-build}"
    
    case "$command" in
        clean)
            clean
            ;;
        test)
            run_tests
            ;;
        example)
            run_example
            ;;
        install)
            install_project
            ;;
        help|--help|-h)
            show_help
            ;;
        build|"")
            print_info "Starting full build process..."
            check_dependencies
            setup_conan
            install_dependencies
            configure
            build
            print_info "Build complete! Run './build.sh test' to run tests."
            print_info "Run './build.sh example' to see the demo."
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"