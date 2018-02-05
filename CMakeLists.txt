cmake_minimum_required(VERSION 3.1)
project(colored_sphere_detector)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS 
roscpp
rgbd_drivers_uncc
)
find_package(OpenCV 3 REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS rgbd_drivers_uncc
    DEPENDS OpenCV
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(/opt/ros/kinetic/lib) # needed to find OpenNI2Orbbec

add_executable(${PROJECT_NAME}_bagfile 
    src/bagfile.cpp
)

target_link_libraries(${PROJECT_NAME}_bagfile
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBRARIES}
    ${rgbd_drivers_uncc_LIBRARIES}
)

add_executable(${PROJECT_NAME}_live 
    src/live.cpp
)

target_link_libraries(${PROJECT_NAME}_live
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBRARIES}
    ${rgbd_drivers_uncc_LIBRARIES}
)