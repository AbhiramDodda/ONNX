cmake_minimum_required(VERSION 3.15)
project(MiniONNXRuntime VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Dependencies
find_package(nlohmann_json REQUIRED)

# Library
add_library(mini_onnx_runtime
    src/model.cpp
    src/graph.cpp
    src/runtime.cpp
    src/kernels.cpp
)

target_include_directories(mini_onnx_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_onnx_runtime
    PUBLIC
        nlohmann_json::nlohmann_json
)

# Main executable
if(BUILD_EXAMPLES)
    add_executable(mini_onnx_main src/main.cpp)
    target_link_libraries(mini_onnx_main PRIVATE mini_onnx_runtime)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    
    if(Catch2_FOUND)
        add_executable(test_runtime tests/test_runtime.cpp)
        target_link_libraries(test_runtime PRIVATE mini_onnx_runtime Catch2::Catch2WithMain)
        
        include(CTest)
        include(Catch)
        catch_discover_tests(test_runtime)
    else()
        message(WARNING "Catch2 not found, tests will not be built")
    endif()
endif()

# Install
install(TARGETS mini_onnx_runtime
    EXPORT MiniONNXRuntimeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)