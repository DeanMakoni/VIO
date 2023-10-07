
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
project(Tutorial)

#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

#find_package(GLog REQUIRED)
#include_directories(${GLog_INCLUDE_DIRS})

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIRS})

find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIRS})

find_package(Boost REQUIRED filesystem)
include_directories(${Boost_INCLUDE_DIRS})

INCLUDE(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})



set(${PROJECT_NAME}_srcs
        src/vio.cpp
       # src/system.cpp
       # include/aru/core/vio/vio.h
       # include/aru/core/vio/system.h
        )

set(${PROJECT_NAME}_ext_libs
       # ${OpenCV_LIBS}
        Eigen3::Eigen
        gflags
        #glog::glog
        gtsam
        ${Boost_LIBRARIES}
        )

add_executable(Tutorial src/vio.cpp)
target_link_libraries(${PROJECT_NAME}
        ${${PROJECT_NAME}_ext_libs})
