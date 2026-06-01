cmake_minimum_required(VERSION 3.18)
project(handwriting_cpp_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(bench_train_step bench_train_step.cpp)
target_link_libraries(bench_train_step "${TORCH_LIBRARIES}")
target_compile_options(bench_train_step PRIVATE -O3)

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
