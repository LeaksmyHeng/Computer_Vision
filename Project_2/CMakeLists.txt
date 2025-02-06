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

# # Passing the argument to run the first task
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.1016.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus baseline SSD 5
#     COMMENT "Running the target image with directory path"
# )

# # Passing the argument to run the second task
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.0164.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus histogram histogramIntersection 5
#     COMMENT "Running the target image with directory path"
# )

# # Passing the argument to run the third task
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.0274.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus multiHistogram histogramIntersection 5
#     COMMENT "Running the target image with directory path"
# )

# Passing the argument to run the third task
add_custom_command(TARGET main POST_BUILD
    COMMAND main ${CMAKE_SOURCE_DIR}/olympus/test/pic.0009.jpg ${CMAKE_SOURCE_DIR}/olympus/test multiHistogram histogramIntersection 3
    COMMENT "Running the target image with directory path"
)

# Passing the argument to run the second task
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.0164.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus histogram histogramIntersection 5
#     COMMENT "Running the target image with directory path"
# )

include(CPack)
