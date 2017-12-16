cmake_minimum_required(VERSION 3.5)

project(Caffe_CycleGAN)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

#Add Caffe
find_package(OpenCV)
find_package(Caffe)

#Add source files
file(GLOB src_files "src/*.cpp")
file(GLOB header_files "include/*.h")

include_directories("include")
include_directories(${CAFFE_INCLUDE_DIRS})

add_executable(Caffe_CycleGAN ${src_files} ${header_files})

target_link_libraries(Caffe_CycleGAN ${Caffe_LIBRARIES})