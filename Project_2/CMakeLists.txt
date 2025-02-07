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

add_executable(main main.cpp features.cpp distance_metric.cpp csv_util.cpp)
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

# # Passing the argument to run the fourth task
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.0535.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus colorTexture weightedDistance 5
#     COMMENT "Running the target image with directory path"
# )

# Passing the argument to run the fifth task 0164 & 0893
add_custom_command(TARGET main POST_BUILD
    COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.0164.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus DNN SSD_V 3
    COMMENT "Running the target image with directory path"
)

# # Passing the argument to run the sixth task
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.0287.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus DNN SSD_V 15
#     COMMENT "Running the target image with directory path"
# )
# # Compare with baseline method using SSD
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.1072.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus baseline SSD 3
#     COMMENT "Running the target image with directory path"
# )
# # Compare with baseline method using histogram histogram intersection
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.0734.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus histogram histogramIntersection 3
#     COMMENT "Running the target image with directory path"
# )

# # Passing the argument to run the last task 0280 & the 0343
# add_custom_command(TARGET main POST_BUILD
#     COMMAND main ${CMAKE_SOURCE_DIR}/olympus/olympus/pic.0280.jpg ${CMAKE_SOURCE_DIR}/olympus/olympus CBIR CBIR 5
#     COMMENT "Running the target image with directory path"
# )

include(CPack)
