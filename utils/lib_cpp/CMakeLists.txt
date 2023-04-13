cmake_minimum_required(VERSION 3.5)
project(devkit_object)

set(CMAKE_BUILD_TYPE "Release")
find_package(Boost)
include_directories(${Boost_INCLUDE_DIR})
find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
find_package(OpenCV REQUIRED) 
include_directories(${OpenCV_INCLUDE_DIRS})

set(Boost_USE_MULTITHREAD ON)

add_subdirectory(pybind11)
pybind11_add_module(lib_cpp lib_cpp.cpp)
set_target_properties(lib_cpp PROPERTIES PREFIX "")
set_target_properties(lib_cpp PROPERTIES OUTPUT_NAME "lib_cpp")
set_target_properties(lib_cpp PROPERTIES SUFFIX ".so")

pybind11_add_module(preprocess preprocess.cpp)
set_target_properties(preprocess PROPERTIES PREFIX "")
set_target_properties(preprocess PROPERTIES OUTPUT_NAME "preprocess")
set_target_properties(preprocess PROPERTIES SUFFIX ".so")

pybind11_add_module(postprocess postprocess.cpp)
set_target_properties(postprocess PROPERTIES PREFIX "")
set_target_properties(postprocess PROPERTIES OUTPUT_NAME "postprocess")
set_target_properties(postprocess PROPERTIES SUFFIX ".so")
target_link_libraries(postprocess PRIVATE ${OpenCV_LIBS})



