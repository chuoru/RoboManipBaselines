cmake_minimum_required(VERSION 3.10)
project(insta360_bridge)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MUST match ORB-SLAM3's own Release flags. Its CMakeLists.txt does
#   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
# which changes Eigen's EIGEN_MAX_ALIGN_BYTES (16 for baseline x86-64/SSE2, 32
# once AVX is enabled). Any struct containing a fixed-size vectorizable Eigen
# type -- Sophus::SE3f, IMU::Point, Eigen::Matrix4f, ... -- then has a
# different alignment (and padding) in this binary than in libORB_SLAM3.so.
# That is an ODR violation: objects allocated inside the library and freed here
# (or vice versa) corrupt the heap.
#
# Observed symptom before this was added, on real hardware: tracking initialised
# fine ("New Map created with 352 points") and the process immediately aborted
# with "double free or corruption (out)" -- reproducibly, at the exact moment
# monocular-inertial initialisation completed and Eigen-heavy objects started
# crossing the library boundary in bulk.
#
# It is deliberately EMPTY by default, because Dockerfile.insta360 now strips
# -march=native from ORB-SLAM3/DBoW2/g2o as well. Reason: -march=native enables
# AVX, which raises Eigen's alignment requirement for fixed-size vectorizable
# types from 16 to 32 bytes, and the vendored 2011-era g2o does not apply
# EIGEN_MAKE_ALIGNED_OPERATOR_NEW consistently to its graph vertices/edges.
# At 16 bytes glibc's natural malloc alignment covers it; at 32 it does not,
# and the heap corruption surfaces as "double free or corruption (out)" inside
# g2o::HyperGraph::clear() when the first global bundle adjustment tears the
# optimizer down. See that Dockerfile's comment for the full gdb backtrace.
#
# If you ever set this back to -march=native, you MUST rebuild ORB-SLAM3 and
# both of its Thirdparty libraries with the same flag -- and expect the g2o
# crash above to return.
set(INSTA360_BRIDGE_ARCH_FLAGS "" CACHE STRING
    "Arch flags; MUST match those ORB-SLAM3 was built with (Eigen ABI)")
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 ${INSTA360_BRIDGE_ARCH_FLAGS}")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
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
# Overridable, matching the INSTA360_SDK_DIR pattern above: the Docker image
# (Dockerfile.insta360) builds ORB-SLAM3 into /opt/ORB_SLAM3 rather than the
# submodule, and passes -DORB_SLAM3_DIR. A plain set() here would silently
# discard that.
if(NOT ORB_SLAM3_DIR)
  set(ORB_SLAM3_DIR "$ENV{ORB_SLAM3_DIR}")
endif()
set(ORB_SLAM3_DIR "${ORB_SLAM3_DIR}" CACHE PATH
    "Root of a built ORB-SLAM3 tree (contains lib/libORB_SLAM3.so)")
if(NOT ORB_SLAM3_DIR)
  set(ORB_SLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/ORB_SLAM3)
endif()
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

# On by default: verified against real hardware (Insta360 X3, fw v1.0.83,
# SDK 2.1.8) that OnGyroData fires with real accel+gyro data when the camera
# is physically set to dual-lens (360) capture mode via its own screen -- see
# main.cc's INSTA360_GYRO_HAS_ACCEL comment. `-DINSTA360_GYRO_HAS_ACCEL=OFF`
# forces the IMU-less fallback.
option(INSTA360_GYRO_HAS_ACCEL "GyroData includes accelerometer samples" ON)
target_compile_definitions(insta360_bridge PRIVATE
  INSTA360_GYRO_HAS_ACCEL=$<BOOL:${INSTA360_GYRO_HAS_ACCEL}>)

target_include_directories(insta360_bridge PRIVATE
  ${INSTA360_SDK_DIR}/include
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
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
  ${Pangolin_LIBRARIES}
  PkgConfig::AVCODEC
  PkgConfig::AVUTIL
  PkgConfig::SWSCALE
  Threads::Threads
)
