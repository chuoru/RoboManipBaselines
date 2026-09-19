// insta360_bridge: connects to an Insta360 X3/X4 over USB via the Insta360
// CameraSDK, decodes its preview video stream, feeds video + gyro into
// ORB-SLAM3 (Monocular-Inertial), and publishes both the decoded RGB frames
// and the estimated 6-DoF pose to any number of local clients over a Unix
// domain socket (see Insta360BridgeProtocol.h / common/utils/Insta360Protocol.py
// for the wire format this must match).
//
// See README.md in this directory for build instructions (SDK application,
// ORB-SLAM3 submodule build, udev/permissions, camera USB mode switch) and
// for the KNOWN UNRESOLVED RISK this file's IMU handling depends on: whether
// ins_camera::GyroData carries accelerometer samples at all (the public SDK
// README only documents gyro). See the INSTA360_GYRO_HAS_ACCEL block below.
//
// Usage:
//   ./insta360_bridge --socket /tmp/insta360_hand.sock \
//       --vocab /path/to/ORBvoc.txt --settings ./Insta360_X4.yaml
//
// Against misc/MockInsta360Bridge.py instead of this real bridge, the exact
// same socket path lets the Python side and MuJoCo teleop be exercised
// without any of this file's hardware/SDK dependencies.

#include <camera/camera.h>
#include <camera/device_discovery.h>
#include <camera/photography_settings.h>

#include <System.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "BridgeServer.h"
#include "Insta360BridgeProtocol.h"
#include "StreamDecoder.h"

// Set at configure time (e.g. `cmake -DINSTA360_GYRO_HAS_ACCEL=ON`) only
// once confirmed against the real SDK header -- see README.md. Until then
// this file defaults to the IMU-less fallback: ORB_SLAM3::System::MONOCULAR
// instead of ::IMU_MONOCULAR, with pos_scale (Insta360UMI.yaml, same field
// ViveInputDevice already uses) absorbing the resulting scale ambiguity.
#ifndef INSTA360_GYRO_HAS_ACCEL
#define INSTA360_GYRO_HAS_ACCEL 0
#endif

namespace {

std::atomic<bool> g_running{true};

void SignalHandler(int) { g_running = false; }

struct ImuSample {
  double timestamp_sec;
  float gyro[3];
  float accel[3];
};

// Bridges the CameraSDK's async StreamDelegate callbacks into simple
// queues the main loop below can drain each iteration -- same reasoning as
// RealEnvBase's various background-thread/queue sensor setups (e.g.
// setup_femtobolt, setup_m5stack_scale): OnVideoData/OnGyroData run on an
// SDK-owned thread and must never block.
class BridgeStreamDelegate : public ins_camera::StreamDelegate {
 public:
  explicit BridgeStreamDelegate(bool is_h265) : decoder_(is_h265) {}

  void OnAudioData(const uint8_t*, size_t, int64_t) override {}

  void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp,
                    uint8_t /*stream_type*/, int stream_index) override {
    // README: below 5.7K (our preview resolution is 1920x960) the stream is
    // a single video stream, so only stream_index 0 is ever populated.
    if (stream_index != 0) {
      return;
    }
    cv::Mat frame;
    if (decoder_.Decode(data, size, frame)) {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      latest_frame_ = frame;
      // CameraSDK timestamps: unit not documented in the public README:
      // treated here as microseconds (matching typical SDK convention);
      // confirm against the real header and adjust FRAME_TIMESTAMP_SCALE
      // below if wrong. Only relative spacing between frame/IMU timestamps
      // actually matters for TrackMonocular, not the absolute epoch.
      latest_frame_timestamp_sec_ = static_cast<double>(timestamp) * 1e-6;
      has_frame_ = true;
    }
  }

  void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    for (const auto& sample : data) {
      ImuSample imu_sample{};
      // TODO(insta360-sdk): confirm these field names and units against the
      // real camera/camera.h once obtained -- see README.md's "Known
      // unresolved risk". gyro_x/y/z [rad/s] assumed here.
      imu_sample.timestamp_sec = static_cast<double>(sample.timestamp) * 1e-6;
      imu_sample.gyro[0] = sample.gyro_x;
      imu_sample.gyro[1] = sample.gyro_y;
      imu_sample.gyro[2] = sample.gyro_z;
#if INSTA360_GYRO_HAS_ACCEL
      imu_sample.accel[0] = sample.accel_x;
      imu_sample.accel[1] = sample.accel_y;
      imu_sample.accel[2] = sample.accel_z;
#else
      imu_sample.accel[0] = imu_sample.accel[1] = imu_sample.accel[2] = 0.0f;
#endif
      imu_queue_.push_back(imu_sample);
    }
  }

  void OnExposureData(const ins_camera::ExposureData&) override {}

  // Pops the latest decoded frame, if a new one has arrived since the last
  // call. Returns false if none is available yet.
  bool TakeLatestFrame(cv::Mat& frame, double& timestamp_sec) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!has_frame_) {
      return false;
    }
    frame = latest_frame_;
    timestamp_sec = latest_frame_timestamp_sec_;
    has_frame_ = false;
    return true;
  }

  // Drains every IMU sample queued since the last call.
  std::vector<ImuSample> TakeImuSamples() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    std::vector<ImuSample> samples(imu_queue_.begin(), imu_queue_.end());
    imu_queue_.clear();
    return samples;
  }

 private:
  StreamDecoder decoder_;

  std::mutex frame_mutex_;
  cv::Mat latest_frame_;
  double latest_frame_timestamp_sec_ = 0.0;
  bool has_frame_ = false;

  std::mutex imu_mutex_;
  std::deque<ImuSample> imu_queue_;
};

std::string TrackingStateToString(int state) {
  // ORB_SLAM3::Tracking::eTrackingState: SYSTEM_NOT_READY/NO_IMAGES_YET/
  // NOT_INITIALIZED map to "INIT", OK to "OK", everything else (LOST,
  // RECENTLY_LOST) to "LOST" -- see Insta360InputDevice.py, which only ever
  // acts on "OK".
  switch (state) {
    case 2:  // ORB_SLAM3::Tracking::OK
      return "OK";
    case -1:
    case 0:
    case 1:
      return "INIT";
    default:
      return "LOST";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  std::string socket_path;
  std::string vocab_path;
  std::string settings_path;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--socket" && i + 1 < argc) {
      socket_path = argv[++i];
    } else if (arg == "--vocab" && i + 1 < argc) {
      vocab_path = argv[++i];
    } else if (arg == "--settings" && i + 1 < argc) {
      settings_path = argv[++i];
    }
  }
  if (socket_path.empty() || vocab_path.empty() || settings_path.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --socket <path> --vocab <ORBvoc.txt> --settings <yaml>"
              << std::endl;
    return 1;
  }

  // --- Connect to the camera (ins_camera::DeviceDiscovery/Camera/Open --
  // exact API confirmed against the public README/demo, see that file) ---
  ins_camera::DeviceDiscovery discovery;
  auto device_list = discovery.GetAvailableDevices();
  if (device_list.empty()) {
    std::cerr << "[insta360_bridge] No Insta360 camera found. Is it "
                 "connected and switched to Android USB mode? See "
                 "README.md."
              << std::endl;
    return 1;
  }
  auto camera = std::make_shared<ins_camera::Camera>(device_list[0].info);
  if (!camera->Open()) {
    std::cerr << "[insta360_bridge] Failed to open camera." << std::endl;
    return 1;
  }
  discovery.FreeDeviceDescriptors(device_list);
  std::cout << "[insta360_bridge] Camera opened." << std::endl;

  // TODO(insta360-sdk): GetVideoEncodeType()'s return type/enum values
  // (VideoEncodeType::H265 assumed here) are not shown in the public
  // README -- confirm the exact type/enumerators against the real header
  // and adjust. Same category of unknown as GyroData's fields above.
  bool is_h265 = camera->GetVideoEncodeType() == ins_camera::VideoEncodeType::H265;
  auto delegate = std::make_shared<BridgeStreamDelegate>(is_h265);
  camera->SetStreamDelegate(delegate);

  ins_camera::LiveStreamParam stream_param;
  // README's recommended preview resolution -- also keeps the video single-
  // stream (< 5.7K), see BridgeStreamDelegate::OnVideoData.
  stream_param.video_resolution = ins_camera::VideoResolution::RES_1920_960P30;
  stream_param.lrv_video_resulution = ins_camera::VideoResolution::RES_1920_960P30;
  stream_param.video_bitrate = 1024 * 1024 / 2;
  stream_param.enable_audio = false;
  stream_param.using_lrv = false;
  if (!camera->StartLiveStreaming(stream_param)) {
    std::cerr << "[insta360_bridge] Failed to start live streaming."
              << std::endl;
    return 1;
  }
  std::cout << "[insta360_bridge] Live streaming started." << std::endl;

  // --- ORB-SLAM3 ---
#if INSTA360_GYRO_HAS_ACCEL
  const auto sensor_mode = ORB_SLAM3::System::IMU_MONOCULAR;
  std::cout << "[insta360_bridge] Running ORB-SLAM3 in IMU_MONOCULAR mode."
            << std::endl;
#else
  const auto sensor_mode = ORB_SLAM3::System::MONOCULAR;
  std::cout << "[insta360_bridge] INSTA360_GYRO_HAS_ACCEL is not set -- "
               "running ORB-SLAM3 in MONOCULAR (no IMU) mode. Scale is "
               "then ambiguous; tune Insta360UMI.yaml's pos_scale by hand. "
               "See README.md's \"Known unresolved risk\"."
            << std::endl;
#endif
  ORB_SLAM3::System slam(vocab_path, settings_path, sensor_mode,
                          /*use_viewer=*/false);

  BridgeServer server(socket_path);
  server.Start();
  std::cout << "[insta360_bridge] Serving on " << socket_path << std::endl;

  std::vector<ORB_SLAM3::IMU::Point> imu_meas;
  while (g_running) {
    cv::Mat frame;
    double frame_timestamp_sec;
    if (!delegate->TakeLatestFrame(frame, frame_timestamp_sec)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    // Publish the frame to any "frame"-consuming clients (RealEnvBase.
    // setup_insta360) regardless of tracking state -- the recorded camera
    // image is useful even while SLAM is still initializing/relocalizing.
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    server.Broadcast(EncodeFrameMessage(frame_timestamp_sec, rgb_frame.cols,
                                         rgb_frame.rows, rgb_frame.data));

    for (const auto& sample : delegate->TakeImuSamples()) {
      imu_meas.emplace_back(sample.accel[0], sample.accel[1], sample.accel[2],
                             sample.gyro[0], sample.gyro[1], sample.gyro[2],
                             sample.timestamp_sec);
    }

    Sophus::SE3f Tcw =
        slam.TrackMonocular(frame, frame_timestamp_sec, imu_meas);
    imu_meas.clear();

    Sophus::SE3f Twc = Tcw.inverse();
    Eigen::Vector3f t = Twc.translation();
    Eigen::Quaternionf q = Twc.unit_quaternion();

    std::string tracking_state = TrackingStateToString(slam.GetTrackingState());
    server.Broadcast(EncodePoseMessage(frame_timestamp_sec, t.x(), t.y(),
                                        t.z(), q.w(), q.x(), q.y(), q.z(),
                                        tracking_state));
  }

  std::cout << "[insta360_bridge] Shutting down." << std::endl;
  camera->StopLiveStreaming();
  camera->Close();
  slam.Shutdown();
  server.Stop();
  return 0;
}
