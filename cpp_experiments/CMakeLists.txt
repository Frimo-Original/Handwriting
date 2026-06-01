cmake_minimum_required(VERSION 3.18)
project(handwriting_cpp_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Newer CUDA installations may not provide the legacy nvToolsExt CMake target
# that PyTorch's Caffe2 config still references on Windows.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND AND NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(bench_train_step bench_train_step.cpp)
target_link_libraries(bench_train_step "${TORCH_LIBRARIES}")

if(CUDAToolkit_FOUND)
  target_compile_definitions(bench_train_step PRIVATE USE_CUDA)
endif()

if(MSVC)
  target_compile_options(bench_train_step PRIVATE /O2 /EHsc)
else()
  target_compile_options(bench_train_step PRIVATE -O3)
endif()

if(APPLE)
  set_target_properties(
    bench_train_step
    PROPERTIES
      BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )
endif()

if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(
    TARGET bench_train_step
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:bench_train_step>
  )
endif()
