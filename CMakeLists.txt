cmake_minimum_required(VERSION 3.0)

project(cropper LANGUAGES CXX CUDA)

# find_package(ffmpeg REQUIRED)
find_package(SDL2 REQUIRED)
# Add the NATS library
find_library(NATS_LIBRARY nats PATHS /usr/local/lib)

if (NOT NATS_LIBRARY)
    message(FATAL_ERROR "NATS library not found")
endif()
enable_language(CUDA)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${SDL2_INCLUDE_DIRS})
add_executable(cropper src/main.cu)
set_property(TARGET cropper PROPERTY CXX_STANDARD 14 CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cropper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cropper ${SDL2_LIBRARIES} ${NATS_LIBRARY})
