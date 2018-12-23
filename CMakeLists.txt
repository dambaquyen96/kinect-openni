cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

project(openni_grabber)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# install library OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/include/hiredis)

add_executable (openni_grabber dectect_object.cpp DeterminePosition.h Location.h)
target_link_libraries (openni_grabber ${PCL_LIBRARIES})
target_link_libraries(openni_grabber ${OpenCV_LIBS} )
target_link_libraries(openni_grabber /usr/local/lib/libhiredis.so)
