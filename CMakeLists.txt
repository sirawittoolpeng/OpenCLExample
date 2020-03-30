cmake_minimum_required(VERSION 2.8.3)
project(OpenCLExample)
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(inv src/inv.cpp)
target_link_libraries(inv ${OpenCV_LIBS} ${GPERFTOOLS_LIBRARIES}) 

add_executable(vect src/vector.cpp)
target_link_libraries(vect ${OpenCL_LIBRARY} ${GPERFTOOLS_LIBRARIES})