cmake_minimum_required(VERSION 2.8.3)
project(modelify_tools)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_executable(align_objects
  src/align_objects.cc
)

cs_add_executable(modify_ply_files
  src/modify_ply_files.cc
)

cs_install()
cs_export()
