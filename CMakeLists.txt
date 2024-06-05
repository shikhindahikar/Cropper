cmake_minimum_required(VERSION 3.0)

project(cropper)

# find_package(ffmpeg REQUIRED)
find_package(SDL2 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${SDL2_INCLUDE_DIRS})
add_executable(cropper src/main.cpp)
target_include_directories(cropper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cropper ${SDL2_LIBRARIES})
