cmake_minimum_required(VERSION 3.10.0)
project(project_2 VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CTest)
enable_testing()

add_executable(main main.cpp features.cpp distance_metric.cpp)
target_link_libraries(main ${OpenCV_LIBS})

# Passing the argument to run the timeBlue.cpp
add_custom_command(TARGET main POST_BUILD
    COMMAND main ${CMAKE_SOURCE_DIR}/olympus/test/pic.0009.jpg ${CMAKE_SOURCE_DIR}/olympus/test baseline SSD
    COMMENT "Running the target image with directory path"
)


include(CPack)
