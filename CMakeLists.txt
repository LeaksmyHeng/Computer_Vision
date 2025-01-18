cmake_minimum_required(VERSION 3.10.0)
project(project_1 VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(project_1 imgDisplay.cpp)
add_executable(project_1 vidDisplay.cpp filter.cpp)
target_link_libraries(project_1 ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
