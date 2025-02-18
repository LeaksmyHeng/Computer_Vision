cmake_minimum_required(VERSION 3.10.0)
project(project_3 VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

add_executable(project_3 main.cpp Common/thresholding.cpp Common/morphological.cpp Common/segmentation.cpp)
target_link_libraries(project_3 ${OpenCV_LIBS})

include(CTest)
include(CPack)
enable_testing()
