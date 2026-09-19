cmake_minimum_required(VERSION 3.10)
project(insta360_bridge)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)
pkg_check_modules(SWSCALE REQUIRED IMPORTED_TARGET libswscale)

# --- Insta360 CameraSDK ------------------------------------------------
# Not redistributable: apply for it at https://www.insta360.com/sdk/apply
# and point this at wherever you unpacked it (must contain include/camera/
# and lib/). See README.md.
if(NOT INSTA360_SDK_DIR)
  set(INSTA360_SDK_DIR "$ENV{INSTA360_SDK_DIR}")
endif()
if(NOT INSTA360_SDK_DIR)
  message(FATAL_ERROR
    "INSTA360_SDK_DIR is not set. Pass -DINSTA360_SDK_DIR=/path/to/sdk or "
    "set the INSTA360_SDK_DIR environment variable -- see README.md.")
endif()

# --- ORB-SLAM3 -----------------------------------------------------------
# Vendored as ../../../third_party/ORB_SLAM3 (git submodule). Build it
# separately first per its own README (produces lib/libORB_SLAM3.so and its
# bundled Thirdparty/{DBoW2,g2o}/lib/*.so) -- this project only links
# against that already-built output, it does not build ORB-SLAM3 itself.
set(ORB_SLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ORB_SLAM3)
if(NOT EXISTS ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so)
  message(FATAL_ERROR
    "third_party/ORB_SLAM3 has not been built yet (expected "
    "${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so). Follow that submodule's own "
    "README/build.sh first.")
endif()

add_executable(insta360_bridge
  main.cc
  StreamDecoder.cc
  BridgeServer.cc
)

# Off by default: only enable once confirmed against the real SDK header
# that ins_camera::GyroData actually carries accelerometer samples -- see
# README.md's "Known unresolved risk" and main.cc's INSTA360_GYRO_HAS_ACCEL
# usage. `cmake -B build -DINSTA360_GYRO_HAS_ACCEL=ON` to turn it on.
option(INSTA360_GYRO_HAS_ACCEL "GyroData includes accelerometer samples" OFF)
if(INSTA360_GYRO_HAS_ACCEL)
  target_compile_definitions(insta360_bridge PRIVATE INSTA360_GYRO_HAS_ACCEL=1)
endif()

target_include_directories(insta360_bridge PRIVATE
  ${INSTA360_SDK_DIR}/include
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(insta360_bridge PRIVATE
  ${INSTA360_SDK_DIR}/lib
  ${ORB_SLAM3_DIR}/lib
  ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib
  ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib
)

# CameraSDK's exact library name(s) depend on the applied-for package
# contents (observed as libCameraSDK.so in third-party integration reports;
# confirm against your own SDK's lib/ directory and adjust if needed).
target_link_libraries(insta360_bridge PRIVATE
  CameraSDK
  ORB_SLAM3
  DBoW2
  g2o
  ${OpenCV_LIBS}
  PkgConfig::AVCODEC
  PkgConfig::AVUTIL
  PkgConfig::SWSCALE
  Threads::Threads
)
