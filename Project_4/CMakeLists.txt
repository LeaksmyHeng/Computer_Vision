cmake_minimum_required(VERSION 3.10.0)
project(project_4 VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

# uncomment this if you want to run task 1 to 3
# add_executable(project_4 main.cpp Utils/image_detection.cpp Utils/utils.cpp)
# this is for task 4
add_executable(project_4 camera_pos.cpp Utils/image_detection.cpp Utils/utils.cpp)
# add_executable(project_4 harris_corner.cpp Utils/image_detection.cpp Utils/utils.cpp)
target_link_libraries(project_4 ${OpenCV_LIBS})

include(CTest)
include(CPack)
enable_testing()
