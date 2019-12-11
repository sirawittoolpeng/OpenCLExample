cmake_minimum_required(VERSION 2.8.3)
project(OpenCLExample)
find_package(OpenCV REQUIRED)
add_executable(oclgb src/gaussianblur.cpp)
target_link_libraries(oclgb ${OpenCV_LIBS})

add_executable(shift src/shift.cpp)
target_link_libraries(shift ${OpenCV_LIBS})

add_executable(negaposi src/negaposi.cpp)
target_link_libraries(negaposi ${OpenCV_LIBS})

add_executable(negaposi2 src/negaposi2.cpp)
target_link_libraries(negaposi2 ${OpenCV_LIBS})