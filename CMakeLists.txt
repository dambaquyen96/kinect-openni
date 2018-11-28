cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(openni_grabber)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (openni_grabber dectect_object.cpp DeterminePosition.h)
target_link_libraries (openni_grabber ${PCL_LIBRARIES})
