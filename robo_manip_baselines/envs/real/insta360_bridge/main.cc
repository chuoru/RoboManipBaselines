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
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "BridgeServer.h"
#include "Insta360BridgeProtocol.h"
#include "StreamDecoder.h"

// Confirmed against real hardware (Insta360 X3, firmware v1.0.83, SDK
// 2.1.8) that ins_camera::GyroData (stream/stream_types.h) carries both
// accelerometer (ax/ay/az, in units of g -- see the *9.80665 conversion in
// OnGyroData below) and gyro (gx/gy/gz, rad/s) samples, and that
// StreamDelegate::OnGyroData fires with real data -- but only when the
// camera is physically set to dual-lens ("360") capture mode via its own
// screen (Settings -> General -> Camera Mode on this X3), not something
// forced from software; it does not fire in single-lens mode. Pass
// `cmake -DINSTA360_GYRO_HAS_ACCEL=OFF` to force the old IMU-less
// MONOCULAR fallback instead.
#ifndef INSTA360_GYRO_HAS_ACCEL
#define INSTA360_GYRO_HAS_ACCEL 1
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
//
// OnVideoData/OnGyroData are both dispatched from that SAME SDK-owned
// thread (confirmed against real hardware: running IMU_MONOCULAR SLAM +
// viewer live produced long runs -- tens of consecutive frames -- of
// "Empty IMU measurements vector!!!" that never happened during a plain
// --no-slam --record capture of the same camera). decoder_.Decode() is a
// real ffmpeg H.264/H.265 decode of a 3840x1920 frame, not free -- under
// the extra CPU/memory-bandwidth pressure from SLAM+viewer running
// concurrently, a slow decode call delays whatever OnGyroData call is
// queued behind it on that one thread, starving ORB_SLAM3's IMU
// preintegration and repeatedly forcing "IMU is not or recently
// initialized. Reseting active map...", which then crashed
// (segfault in libc.so.6, confirmed via dmesg) after enough rapid resets.
// Fix: OnVideoData now only copies the raw compressed bytes into a queue
// (fast, bounded work) and returns; a dedicated decode_thread_ does the
// actual decoding, keeping the SDK thread free to deliver gyro batches
// promptly. decoder_.Decode() is a STATEFUL streaming decoder (fed
// incrementally, not one shot per frame), so chunks must still be fed to
// it strictly in arrival order -- the queue is FIFO and single-consumer,
// preserving that.
class BridgeStreamDelegate : public ins_camera::StreamDelegate {
 public:
  explicit BridgeStreamDelegate(bool is_h265)
      : decoder_(is_h265),
        decode_thread_(&BridgeStreamDelegate::DecodeThreadMain, this) {}

  // Not virtual in the base (ins_camera::StreamDelegate has no virtual
  // destructor) -- safe here since this object is always owned via a
  // shared_ptr<BridgeStreamDelegate> constructed with make_shared, whose
  // deleter is captured for the concrete type, not dispatched virtually.
  ~BridgeStreamDelegate() {
    stop_decode_thread_ = true;
    raw_video_cv_.notify_all();
    decode_thread_.join();
  }

  void OnAudioData(const uint8_t*, size_t, int64_t) override {}

  void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp,
                    uint8_t /*stream_type*/, int stream_index) override {
    // README: below 5.7K (our preview resolution is 1920x960) the stream is
    // a single video stream, so only stream_index 0 is ever populated.
    if (stream_index != 0) {
      return;
    }
    // Local wall-clock receipt time, NOT the SDK's own `timestamp`
    // parameter -- confirmed against real hardware that OnVideoData's
    // `timestamp` and OnGyroData's GyroData::timestamp are on two
    // independent, non-comparable clocks (raw gyro ticks were found
    // sitting ~67.5 SECONDS away from raw video ticks despite both
    // starting at roughly the same real moment, and neither field's
    // documentation states a shared epoch) -- for CROSS-STREAM alignment
    // (what ORB_SLAM3/Basalt need to interleave frames with IMU
    // samples), only a common clock works. Using this bridge process's
    // own receipt time for BOTH frame and IMU data (see
    // OnGyroData/WallClockSeconds below) sidesteps the SDK's ambiguous
    // per-field units/epochs entirely. Captured here (at raw-bytes receipt
    // time), not when decode_thread_ later actually decodes it, since
    // decoding may now be delayed under load and the timestamp must still
    // reflect real capture/arrival time.
    const double receipt_time_sec = WallClockSeconds();
    {
      std::lock_guard<std::mutex> lock(raw_video_mutex_);
      raw_video_queue_.emplace_back(std::vector<uint8_t>(data, data + size),
                                     receipt_time_sec);
    }
    raw_video_cv_.notify_one();
  }

  // Runs on decode_thread_, never on the SDK callback thread. Feeds queued
  // chunks to decoder_ strictly in arrival order.
  void DecodeThreadMain() {
    while (true) {
      std::vector<uint8_t> data;
      double receipt_time_sec;
      {
        std::unique_lock<std::mutex> lock(raw_video_mutex_);
        raw_video_cv_.wait(lock, [this] {
          return stop_decode_thread_ || !raw_video_queue_.empty();
        });
        if (raw_video_queue_.empty()) {
          if (stop_decode_thread_) {
            return;
          }
          continue;
        }
        data = std::move(raw_video_queue_.front().first);
        receipt_time_sec = raw_video_queue_.front().second;
        raw_video_queue_.pop_front();
      }
      cv::Mat frame;
      if (decoder_.Decode(data.data(), data.size(), frame)) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = frame;
        latest_frame_timestamp_sec_ = receipt_time_sec;
        has_frame_ = true;
        has_any_frame_ = true;
        frame_seq_++;
        frame_cv_.notify_one();
      }
    }
  }

  // Seconds since the Unix epoch, wall-clock. Shared time base for BOTH
  // frame and IMU timestamps (see OnVideoData/OnGyroData) -- confirmed
  // against real hardware that the SDK's own OnVideoData `timestamp` and
  // GyroData::timestamp fields are on two independent, non-comparable
  // clocks (raw gyro ticks sat ~67.5 SECONDS away from raw video ticks
  // despite both streams starting at roughly the same real moment, with
  // neither field documenting a shared epoch), which silently broke
  // camera-IMU time alignment for every offline/live IMU_MONOCULAR SLAM
  // test this session -- ORB_SLAM3/Basalt both interleave frames and IMU
  // samples by comparing their timestamps directly, so they must share a
  // real, comparable clock. This process's own receipt time for both
  // guarantees that regardless of what the SDK's internal fields mean.
  static double WallClockSeconds() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    if (data.empty()) {
      return;
    }
    if (first_gyro_wall_clock_sec_ < 0.0) {
      first_gyro_wall_clock_sec_ = WallClockSeconds();
    }
    // The SDK delivers gyro data in bursts, NOT smoothly: confirmed via
    // real hardware measurement (raw inter-batch wall-clock gap, before
    // any of our own interpolation) that every OnGyroData call carries a
    // constant 50 samples and consecutive calls land ~80-170ms apart
    // (~100ms typical) -- i.e. an effective ~10Hz burst-delivery cadence,
    // NOT the "~8 samples per batch" this comment used to (incorrectly)
    // say. The camera's own reported capability metadata separately
    // states gyro_timestamp: 1.6 (a 1.6ms/~625Hz native sample period),
    // so the underlying sensor almost certainly samples smoothly and the
    // SDK just buffers ~50 samples internally before flushing them to us
    // -- a fixed SDK behavior with no exposed config to change it (no
    // batch-size/flush-rate option in the public SDK headers). At 30fps
    // video, most frames land in the 60-90ms gap between bursts and see
    // zero fresh IMU data if driven by frame arrival -- see
    // BridgeStreamDelegate::WaitForNextGyroBatch() and main()'s dedicated
    // slam_thread, which triggers SLAM processing directly from this
    // callback's own notify (via gyro_cv_) rather than on new frames.
    //
    // No per-sample capture time we can trust (see WallClockSeconds'
    // comment) -- interpolate the batch's samples evenly across
    // (previous call's wall-clock receipt time, this call's wall-clock
    // receipt time] so every sample gets a distinct, monotonically
    // increasing time close to its real capture time, comparable with
    // frame timestamps. Feeding ORB_SLAM3 samples with zero-delta
    // timestamps makes Tracking::PreintegrateIMU() divide by zero,
    // producing inf/nan that corrupts IMU initialization (observed via
    // mono_inertial_video: "SO3::exp failed! omega: -nan -nan -nan") --
    // this interpolation also avoids that.
    const double now_sec = WallClockSeconds();
    const double interval_sec = last_gyro_batch_wall_clock_sec_ >= 0.0
                                     ? (now_sec - last_gyro_batch_wall_clock_sec_)
                                     : 0.0;
    const size_t n = data.size();
    for (size_t i = 0; i < n; i++) {
      const auto& sample = data[i];
      ImuSample imu_sample{};
      // Real field names from stream/stream_types.h: gx/gy/gz (gyro, rad/s),
      // ax/ay/az (accel). Accel unit confirmed against
      // ai4ce/insta360_ros_driver (a known-working reference using the same
      // SDK struct): ax/ay/az are in g, not m/s^2 -- that driver multiplies
      // by kStandardGravity before publishing as sensor_msgs/Imu, which
      // ORB_SLAM3::IMU::Point (like ROS IMU messages) expects in m/s^2.
      imu_sample.timestamp_sec =
          (interval_sec > 0.0)
              ? last_gyro_batch_wall_clock_sec_ +
                    interval_sec * (static_cast<double>(i + 1) / n)
              : now_sec;
      imu_sample.gyro[0] = sample.gx;
      imu_sample.gyro[1] = sample.gy;
      imu_sample.gyro[2] = sample.gz;
#if INSTA360_GYRO_HAS_ACCEL
      constexpr double kStandardGravity = 9.80665;
      imu_sample.accel[0] = sample.ax * kStandardGravity;
      imu_sample.accel[1] = sample.ay * kStandardGravity;
      imu_sample.accel[2] = sample.az * kStandardGravity;
#else
      imu_sample.accel[0] = imu_sample.accel[1] = imu_sample.accel[2] = 0.0f;
#endif
      // Sanity-check raw sensor values before they ever enter the
      // pipeline: a corrupted USB packet or sensor glitch feeding a huge
      // or non-finite reading into ORB_SLAM3's preintegration
      // (IMU::Preintegrated::IntegrateNewMeasurement) can, over repeated
      // steps, corrupt its accumulated delta-rotation dR into something
      // NormalizeRotation's SVD can no longer recover a valid orthogonal
      // matrix from (observed via gdb on real hardware as a downstream
      // "Sophus ensure failed... R is not orthogonal" abort, well after
      // the actual bad reading). Limits are generous typical MEMS
      // full-scale ranges (2000 deg/s gyro, 16g accel), not a tight
      // physical model -- this is about rejecting clearly-corrupted
      // packets, not legitimate fast motion. Dropping one sample just
      // means that time slice draws from the previous/next real samples
      // via the usual interpolation, not "no data at all".
      constexpr float kMaxGyroRadPerSec = 34.9f;   // ~2000 deg/s
      constexpr float kMaxAccelMPerSec2 = 156.9f;  // ~16g
      const bool finite_and_plausible =
          std::isfinite(imu_sample.gyro[0]) &&
          std::isfinite(imu_sample.gyro[1]) &&
          std::isfinite(imu_sample.gyro[2]) &&
          std::isfinite(imu_sample.accel[0]) &&
          std::isfinite(imu_sample.accel[1]) &&
          std::isfinite(imu_sample.accel[2]) &&
          std::fabs(imu_sample.gyro[0]) <= kMaxGyroRadPerSec &&
          std::fabs(imu_sample.gyro[1]) <= kMaxGyroRadPerSec &&
          std::fabs(imu_sample.gyro[2]) <= kMaxGyroRadPerSec &&
          std::fabs(imu_sample.accel[0]) <= kMaxAccelMPerSec2 &&
          std::fabs(imu_sample.accel[1]) <= kMaxAccelMPerSec2 &&
          std::fabs(imu_sample.accel[2]) <= kMaxAccelMPerSec2;
      if (!finite_and_plausible) {
        std::cerr << "[insta360_bridge] OnGyroData: dropping implausible "
                      "IMU sample (gyro="
                  << imu_sample.gyro[0] << "," << imu_sample.gyro[1] << ","
                  << imu_sample.gyro[2] << " accel=" << imu_sample.accel[0]
                  << "," << imu_sample.accel[1] << "," << imu_sample.accel[2]
                  << ")" << std::endl;
        continue;
      }
      imu_queue_.push_back(imu_sample);
      // Second, independent queue for --record's CSV writer. It must NOT
      // share imu_queue_ with the SLAM thread: both drains are destructive,
      // so a shared queue means whichever consumer runs first eats the
      // samples and the other sees nothing. See TakeImuSamplesForRecording.
      imu_record_queue_.push_back(imu_sample);
      // Each queue now has exactly one consumer, and a consumer may not be
      // running at all (the SLAM thread is not started for --no-slam, and
      // nothing drains the record queue if the main loop stalls), so cap
      // both. kMaxQueuedImuSamples is ~10s at the measured ~500Hz -- far
      // more than the SLAM thread's normal ~1 gyro-batch (~100ms) of lag,
      // while still bounding memory if a consumer is absent or wedged.
      while (imu_queue_.size() > kMaxQueuedImuSamples) {
        imu_queue_.pop_front();
      }
      while (imu_record_queue_.size() > kMaxQueuedImuSamples) {
        imu_record_queue_.pop_front();
      }
    }
    last_gyro_batch_wall_clock_sec_ = now_sec;
    gyro_batch_seq_++;
    gyro_cv_.notify_one();
  }

  void OnExposureData(const ins_camera::ExposureData&) override {}

  // Pops the latest decoded frame, if a new one has arrived since the last
  // call. Returns false if none is available yet. Used by the main loop's
  // frame-broadcast, where "new since I last checked" is exactly what's
  // wanted.
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

  // Non-consuming: always returns the current latest frame if any has ever
  // arrived, without affecting TakeLatestFrame's "new since last call"
  // bookkeeping. Used by the dedicated SLAM thread, which reads on its own
  // (gyro-batch-driven) cadence independent of the main loop's
  // broadcast-every-frame consumption -- both can read the same frame
  // without racing each other for it.
  bool PeekLatestFrame(cv::Mat& frame, double& timestamp_sec) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!has_any_frame_) {
      return false;
    }
    frame = latest_frame_;
    timestamp_sec = latest_frame_timestamp_sec_;
    return true;
  }

  // Drains every IMU sample queued since the last call. Consumed by the
  // dedicated SLAM thread ONLY -- see TakeImuSamplesForRecording for why
  // the --record writer gets its own queue rather than sharing this one.
  std::vector<ImuSample> TakeImuSamples() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    std::vector<ImuSample> samples(imu_queue_.begin(), imu_queue_.end());
    imu_queue_.clear();
    return samples;
  }

  // Same samples, separate queue, for the main loop's --record CSV writer.
  //
  // These two consumers used to share imu_queue_, and both drains clear it.
  // The main loop calls its drain on EVERY decoded frame (~30fps) whenever
  // need_slam_frame is true -- which includes ordinary live SLAM runs with
  // no --record at all, where it immediately threw the samples away. So in
  // live tracking the main loop was stealing most IMU data from the SLAM
  // thread, which then fed TrackMonocular empty windows ("Empty IMU
  // measurements vector!!!"). Keyframes born from those frames get no
  // preintegration, and LocalMapping::InitializeIMU ->
  // Optimizer::InertialOptimization then dereferences a null
  // mpImuPreintegrated -- observed on real hardware as a SIGSEGV inside
  // IMU::Preintegrated::SetNewBias, a few seconds after tracking reached OK.
  //
  // This also explains why a --no-slam --record capture measured a clean
  // ~500Hz with zero empty frame intervals: with no SLAM thread running,
  // nothing was competing for the queue.
  std::vector<ImuSample> TakeImuSamplesForRecording() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    std::vector<ImuSample> samples(imu_record_queue_.begin(),
                                    imu_record_queue_.end());
    imu_record_queue_.clear();
    return samples;
  }

  // Blocks until a new gyro batch has arrived since last_seen_seq, waking
  // immediately via gyro_cv_ (notified at the end of OnGyroData) rather
  // than polling -- used by the dedicated SLAM thread (see main()) so
  // TrackMonocular fires right when a fresh ~50-sample burst lands (see
  // OnGyroData's comment: the SDK delivers gyro data in bursts roughly
  // every ~100ms, not smoothly), instead of on every decoded video frame
  // (~30fps, which left most frames with zero fresh IMU data purely by
  // polling-timing luck). Re-checks g_running every 200ms so shutdown
  // isn't blocked on gyro data that may have stopped arriving. Also
  // naturally covers the old startup gyro-ramp-up gap (video starts
  // immediately, gyro takes ~1.5-2s): the sequence number just doesn't
  // advance yet, so no separate warmup gate is needed.
  uint64_t WaitForNextGyroBatch(uint64_t last_seen_seq) {
    std::unique_lock<std::mutex> lock(imu_mutex_);
    while (g_running && gyro_batch_seq_ == last_seen_seq) {
      gyro_cv_.wait_for(lock, std::chrono::milliseconds(200));
    }
    return gyro_batch_seq_;
  }

  // True once at least one gyro batch has ever arrived. Used to gate the
  // frame-driven SLAM thread's very first calls: before any gyro data
  // exists at all, ORB_SLAM3's internal IMU queue is empty with nothing
  // to eventually fill it from (unlike the steady-state case
  // WaitForNextFrame's comment describes), so those earliest frames are
  // skipped entirely rather than fed to TrackMonocular.
  bool HasReceivedAnyGyroData() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return gyro_batch_seq_ > 0;
  }

  // The wall-clock time up to which REAL (non-synthetic) IMU coverage is
  // guaranteed to already be queued: OnGyroData interpolates each batch's
  // samples across (previous batch's arrival time, this batch's arrival
  // time], so the LAST sample of the most recent batch always has
  // timestamp == last_gyro_batch_wall_clock_sec_ exactly. A frame is safe
  // to submit to TrackMonocular once its own timestamp is <= this
  // watermark -- see main()'s slam_thread, which buffers frames and only
  // submits ones the watermark has already caught up to, instead of
  // fabricating synthetic samples (tried and reverted -- see that
  // function's comment) or accepting empty preintegration windows.
  // Returns -1 before any gyro data has arrived (see HasReceivedAnyGyroData).
  double GetImuWatermarkSec() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return last_gyro_batch_wall_clock_sec_;
  }

  // Blocks until a new frame has been decoded since last_seen_seq, woken
  // via frame_cv_ (notified at the end of DecodeThreadMain). Used by the
  // dedicated SLAM thread (see main()) to trigger TrackMonocular on every
  // new frame again (~30fps) rather than gating on new gyro bursts
  // (~10Hz, see WaitForNextGyroBatch) -- confirmed via real hardware that
  // the ~100ms inter-frame gap from gyro-gating made "Fail to track local
  // map!" resets worse (larger visual baseline between consecutive
  // tracked frames outweighed the benefit of guaranteed non-empty IMU
  // windows). ORB_SLAM3's own internal IMU queue (mlQueueImuData) is
  // persistent across TrackMonocular calls, not per-call, so a call whose
  // own fresh imu_meas happens to be empty can often still be filled from
  // previously-queued-but-unconsumed samples; only feed TrackMonocular
  // nothing when we've never received a single gyro sample yet (that
  // still needs ImuWarmedUp-style gating -- see the caller).
  uint64_t WaitForNextFrame(uint64_t last_seen_seq) {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    while (g_running && frame_seq_ == last_seen_seq) {
      frame_cv_.wait_for(lock, std::chrono::milliseconds(200));
    }
    return frame_seq_;
  }

 private:
  StreamDecoder decoder_;

  std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  cv::Mat latest_frame_;
  double latest_frame_timestamp_sec_ = 0.0;
  bool has_frame_ = false;
  bool has_any_frame_ = false;
  uint64_t frame_seq_ = 0;

  std::mutex imu_mutex_;
  std::condition_variable gyro_cv_;
  static constexpr size_t kMaxQueuedImuSamples = 5000;  // ~10s at ~500Hz
  std::deque<ImuSample> imu_queue_;          // drained by the SLAM thread
  std::deque<ImuSample> imu_record_queue_;   // drained by --record
  double first_gyro_wall_clock_sec_ = -1.0;
  double last_gyro_batch_wall_clock_sec_ = -1.0;
  uint64_t gyro_batch_seq_ = 0;

  // Raw compressed video bytes queued by OnVideoData (SDK thread, must not
  // block) for decode_thread_ to consume. Declared before decode_thread_
  // so they're fully constructed before that thread starts running.
  std::mutex raw_video_mutex_;
  std::condition_variable raw_video_cv_;
  std::deque<std::pair<std::vector<uint8_t>, double>> raw_video_queue_;
  std::atomic<bool> stop_decode_thread_{false};
  // Must be the last member: its constructor starts DecodeThreadMain()
  // immediately, which touches every member above.
  std::thread decode_thread_;
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

// Bounding-box crop/mask constants for BuildSlamFrame below. Hoisted to
// file scope (was local to main()) so both the main loop (record/preview)
// and the dedicated SLAM thread can call BuildSlamFrame with the same
// values.
//
// Widened from the old 210/1500/640/320 (a 1500x1500 crop out of the full
// 1920x1920 lens circle, downscaled to 640x640, masked to radius 320px --
// confirmed against real hardware as tracking rotation poorly, likely
// from the narrow FOV giving weak visual rotation observability). That
// tight crop existed for two reasons, both since resolved: (1) latency --
// full-resolution ORB extraction/Pangolin display caused stutter, but the
// real cause turned out to be video decode blocking the SDK callback
// thread (see BridgeStreamDelegate's DecodeThreadMain/OnVideoData
// comments), not frame size itself; SLAM now runs on its own thread (see
// WaitForNextFrame), decoupled from that SDK thread regardless of its own
// processing rate. (2) the old OpenCV
// fisheye k1-k4 fit only stayed monotonic (physically valid) to ~47
// degrees incidence -- feeding pixels beyond that segfaulted ORB_SLAM3;
// the current Basalt kb4 fit is well-behaved out to ~120 degrees,
// covering the entire lens circle. Now uses the full 1920x1920 circle,
// scaled up proportionally (not just stretching the same 640px over more
// FOV) to roughly preserve pixel density: 640*(1920/1500) =~ 819,
// rounded to 800. Requires a fresh Camera1.fx/fy/cx/cy/k1-k4 calibration
// at this new crop/resolution -- the existing Insta360_X4.yaml values
// were fit for the old 1500->640 pipeline and do NOT apply here.
constexpr int kBoundingBoxOrigin = 0;    // no margin trim -- full lens circle
constexpr int kBoundingBoxSize = 1920;   // full lens circle diameter
constexpr int kSlamFrameSize = 800;      // must match Camera.width/height
// Re-verified after the real recalibration at this resolution (fx=214.09,
// k1-k4 from calib_sweep_800): that fit's own monotonic (physically
// valid) equidistant range is theta<=103.1deg -- NOT the ~120deg the
// pre-recalibration estimate assumed, since a different calibration fit
// can have a different valid range. 400px (the full inscribed circle)
// would reach ~107deg, past that limit -- confirmed via real hardware
// crashes earlier this session that feeding ORB_SLAM3 pixels beyond the
// monotonic range segfaults it. 370px stays at ~93deg with a real
// margin. Re-verify (see calibration/README.md) if Camera1.k1-k4 changes.
//
// Widened 370 -> 385 as a tracking-robustness experiment. Recomputing the
// current kb4 fit's monotonic range exactly (see calibration/README.md for
// the snippet) puts the limit at theta<=103.08deg, which at fx=214.09
// projects to 398.1px -- so the earlier "~107deg at 400px" estimate was
// slightly pessimistic, but the conclusion that 400px overshoots stands.
//   370px -> 90.06deg  (28.1px margin)
//   385px -> 94.73deg  (13.1px margin, +8.3% usable image area)
//   398px -> 102.55deg (0.1px margin -- do not go here)
// 385 keeps a real margin while recovering FOV, aimed at the "Fail to track
// local map!" resets that dominate live tracking (26 in a 70s run) and keep
// maps too short-lived for LocalMapping::InitializeIMU to ever fire. Revert
// to 370 if the beyond-monotonic-range segfault reappears.
constexpr int kMaskRadiusPx = 385;

// Dual-lens (360) capture mode -- required for IMU delivery, see the
// INSTA360_GYRO_HAS_ACCEL comment above -- delivers RAW DUAL-FISHEYE:
// verified against real hardware that the 3840x1920 frame is two
// side-by-side circular fisheye images (one per lens, each ~1920x1920 with
// black corners), NOT a stitched equirectangular panorama as originally
// assumed. KannalaBrandt8 (Insta360_X4.yaml) models exactly one fisheye
// circle, so crop to a single lens only (front/left by default, back/right
// with --lens back) -- feeding it the full double-wide frame put both
// lenses' geometry under one distortion model centered on the gap between
// them, which can't produce a coherent map. Also masks to the
// calibration's valid angular range: Insta360_X4.yaml's Camera1.k1-k4 fit
// only produces a monotonic (physically valid) equidistant distortion
// curve up to a limited incidence angle -- feeding pixels beyond that
// segfaulted ORB_SLAM3 (libORB_SLAM3.so) during map initialization on real
// hardware. Crop+resize preserves the pinhole/fisheye projection geometry
// exactly (a scale + principal-point shift), matching Insta360_X4.yaml's
// rescaled Camera1.fx/fy/cx/cy -- keep this and that YAML in sync if
// either changes. safe_mask is built lazily on first call and cached in
// the caller-owned cv::Mat passed in; give each independent caller (main
// loop, SLAM thread) its own safe_mask instance rather than sharing one
// across threads.
cv::Mat BuildSlamFrame(const cv::Mat& frame, bool use_back_lens,
                        cv::Mat& safe_mask) {
  int crop_x = use_back_lens ? frame.cols - frame.rows : 0;
  cv::Mat lens_crop = frame(cv::Rect(crop_x, 0, frame.rows, frame.rows));

  cv::Mat bbox_crop = lens_crop(cv::Rect(kBoundingBoxOrigin, kBoundingBoxOrigin,
                                          kBoundingBoxSize, kBoundingBoxSize));
  cv::Mat resized;
  cv::resize(bbox_crop, resized, cv::Size(kSlamFrameSize, kSlamFrameSize), 0,
             0, cv::INTER_AREA);

  if (safe_mask.empty()) {
    safe_mask = cv::Mat::zeros(kSlamFrameSize, kSlamFrameSize, CV_8UC1);
    cv::circle(safe_mask, cv::Point(kSlamFrameSize / 2, kSlamFrameSize / 2),
               kMaskRadiusPx, cv::Scalar(255), -1);
  }
  cv::Mat slam_frame = cv::Mat::zeros(resized.size(), resized.type());
  resized.copyTo(slam_frame, safe_mask);
  return slam_frame;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  std::string socket_path;
  std::string vocab_path;
  std::string settings_path;
  bool use_viewer = false;
  bool use_back_lens = false;
  bool no_slam = false;
  bool no_imu = false;
  bool use_preview = false;
  std::string record_prefix;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--socket" && i + 1 < argc) {
      socket_path = argv[++i];
    } else if (arg == "--vocab" && i + 1 < argc) {
      vocab_path = argv[++i];
    } else if (arg == "--settings" && i + 1 < argc) {
      settings_path = argv[++i];
    } else if (arg == "--viewer") {
      use_viewer = true;
    } else if (arg == "--lens" && i + 1 < argc) {
      std::string lens = argv[++i];
      if (lens == "back") {
        use_back_lens = true;
      } else if (lens != "front") {
        std::cerr << "[insta360_bridge] --lens must be 'front' or 'back'"
                  << std::endl;
        return 1;
      }
    } else if (arg == "--no-slam") {
      no_slam = true;
    } else if (arg == "--no-imu") {
      // Forces plain MONOCULAR tracking (no IMU data fed to ORB_SLAM3 at
      // all), using the SAME dedicated SLAM thread as the normal
      // IMU_MONOCULAR path (just skipping the gyro-warmup gate and the
      // watermark-based frame buffering, since neither is relevant
      // without IMU data) -- for isolating whether reset frequency is an
      // IMU/inertial-optimizer issue or a plain visual-tracking issue.
      // Deliberately does NOT exercise the separate, dormant non-accel
      // (INSTA360_GYRO_HAS_ACCEL=0) build path, which has no verified
      // TrackMonocular call site in this build's current thread layout.
      no_imu = true;
    } else if (arg == "--record" && i + 1 < argc) {
      record_prefix = argv[++i];
    } else if (arg == "--preview") {
      use_preview = true;
    }
  }
  if (socket_path.empty() || (!no_slam && (vocab_path.empty() || settings_path.empty()))) {
    std::cerr << "Usage: " << argv[0]
              << " --socket <path> --vocab <ORBvoc.txt> --settings <yaml> "
                 "[--viewer] [--lens front|back] [--no-imu] [--record <prefix>] [--preview]\n"
              << "   or: " << argv[0]
              << " --socket <path> --no-slam [--lens front|back] "
                 "[--record <prefix>] [--preview]"
              << "   (raw frame broadcast only, no ORB-SLAM3/vocab load -- "
                 "for fast calibration-image capture or recording without "
                 "live-SLAM's frame-rate/freeze issues)"
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

  // Force a fast shutter to reduce motion blur -- verified against real
  // hardware that the default auto-exposure (presumably choosing a slower
  // shutter to keep indoor brightness up) produces visibly blurry frames
  // during camera motion, which also plausibly costs ORB-SLAM3 the sharp
  // keypoints it needs to initialize while moving (the one time parallax
  // is actually available). Trade-off: noisier/darker footage in dim
  // rooms, since ISO isn't raised to compensate (SHUTTER_PRIORITY leaves
  // ISO on auto within VideoISOTopLimit, not fixed).
  {
    auto exposure_settings = std::make_shared<ins_camera::ExposureSettings>();
    exposure_settings->SetExposureMode(
        ins_camera::PhotographyOptions_ExposureMode::SHUTTER_PRIORITY);
    exposure_settings->SetShutterSpeed(1.0 / 120.0);
    if (!camera->SetExposureSettings(
            ins_camera::CameraFunctionMode::FUNCTION_MODE_NORMAL_VIDEO,
            exposure_settings)) {
      std::cerr << "[insta360_bridge] Warning: failed to set fast-shutter "
                   "exposure settings; motion blur may be worse than "
                   "expected."
                << std::endl;
    }
  }

  // TODO(insta360-sdk): GetVideoEncodeType()'s return type/enum values
  // (VideoEncodeType::H265 assumed here) are not shown in the public
  // README -- confirm the exact type/enumerators against the real header
  // and adjust. Same category of unknown as GyroData's fields above.
  bool is_h265 = camera->GetVideoEncodeType() == ins_camera::VideoEncodeType::H265;
  auto delegate = std::make_shared<BridgeStreamDelegate>(is_h265);
  // SetStreamDelegate takes std::shared_ptr<StreamDelegate>& (non-const
  // lvalue ref), so the shared_ptr<BridgeStreamDelegate> -> shared_ptr<
  // StreamDelegate> conversion needs a named variable, not a temporary.
  std::shared_ptr<ins_camera::StreamDelegate> stream_delegate = delegate;
  camera->SetStreamDelegate(stream_delegate);

  // NOTE: does NOT call SetVideoCaptureParams/RecordParams to force a
  // "Sphere-mode" resolution (an earlier version of this file did). Per
  // ai4ce/insta360_ros_driver -- a known-working ROS2 driver using the same
  // CameraSDK struct/callback -- OnGyroData fires fine with a plain
  // StartLiveStreaming(RES_1920_960P30) call and no RecordParams call at
  // all; their README instead says gyro/dual-lens delivery depends on the
  // camera being physically set to "dual-lens mode" via its own menu, not
  // an SDK call. Re-verify against real hardware: if this camera isn't
  // already in dual-lens mode, gyro delivery may need that manual step.
  ins_camera::LiveStreamParam stream_param;
  // README's recommended preview resolution -- also keeps the video single-
  // stream (< 5.7K), see BridgeStreamDelegate::OnVideoData. In practice
  // (dual-lens/360 mode), the camera ignores this and delivers 3840x1920
  // regardless -- see the resolution-instability note in README.md.
  stream_param.video_resolution = ins_camera::VideoResolution::RES_1920_960P30;
  stream_param.lrv_video_resulution = ins_camera::VideoResolution::RES_1920_960P30;
  // 512kbps (1024*1024/2) was sized for the requested-but-not-actually-
  // delivered 1920x960 preview; verified against real hardware that at the
  // real 3840x1920 delivered resolution this is far too low, causing
  // visible blur/blockiness especially under motion. 8Mbps is a starting
  // point for real-time (non-archival) quality at this resolution --
  // increase further if quality still isn't sufficient for calibration.
  stream_param.video_bitrate = 8 * 1024 * 1024;
  stream_param.enable_audio = false;
  // Verified against real hardware that using_lrv=true delivers NO video
  // data at all in dual-lens/360 mode (zero OnVideoData calls, camera
  // still reports StartLiveStreaming success) -- not a usable path to a
  // smaller camera-side stream for this mode. The crop-to-mask-bbox +
  // resize-to-640 pipeline below is the actual resolution-reduction fix.
  stream_param.using_lrv = false;
  stream_param.enable_gyro = true;
  if (!camera->StartLiveStreaming(stream_param)) {
    std::cerr << "[insta360_bridge] Failed to start live streaming."
              << std::endl;
    return 1;
  }
  std::cout << "[insta360_bridge] Live streaming started." << std::endl;

  // --- ORB-SLAM3 (skipped entirely with --no-slam) ---
  std::unique_ptr<ORB_SLAM3::System> slam;
  if (!no_slam) {
#if INSTA360_GYRO_HAS_ACCEL
    const auto sensor_mode = no_imu ? ORB_SLAM3::System::MONOCULAR
                                     : ORB_SLAM3::System::IMU_MONOCULAR;
    if (no_imu) {
      std::cout << "[insta360_bridge] --no-imu: running ORB-SLAM3 in "
                   "MONOCULAR (no IMU) mode. Scale is then ambiguous; tune "
                   "Insta360UMI.yaml's pos_scale by hand. See README.md's "
                   "\"Known unresolved risk\"."
                << std::endl;
    } else {
      std::cout << "[insta360_bridge] Running ORB-SLAM3 in IMU_MONOCULAR mode."
                << std::endl;
    }
#else
    const auto sensor_mode = ORB_SLAM3::System::MONOCULAR;
    std::cout << "[insta360_bridge] INSTA360_GYRO_HAS_ACCEL is not set -- "
                 "running ORB-SLAM3 in MONOCULAR (no IMU) mode. Scale is "
                 "then ambiguous; tune Insta360UMI.yaml's pos_scale by hand. "
                 "See README.md's \"Known unresolved risk\"."
              << std::endl;
#endif
    slam = std::make_unique<ORB_SLAM3::System>(vocab_path, settings_path,
                                                sensor_mode, use_viewer);
  } else {
    std::cout << "[insta360_bridge] --no-slam: broadcasting raw frames "
                 "only, no tracking (no vocab load, much faster startup)."
              << std::endl;
  }

  // Own safe_mask instance for this loop's BuildSlamFrame calls
  // (record/preview) -- the dedicated SLAM thread (see below) builds its
  // own separately, since BuildSlamFrame's mask/crop rationale says each
  // independent caller should have its own rather than sharing one across
  // threads.
  cv::Mat safe_mask;

  const bool need_slam_frame = !no_slam || !record_prefix.empty() || use_preview;

  // --record <prefix>: writes <prefix>.mp4 (the same crop+resize+mask
  // pipeline SLAM itself sees, so a recording made in --no-slam mode --
  // for full, unthrottled frame rate -- can be replayed offline later
  // through the same calibration/mask as the live path, e.g. with
  // ORB_SLAM3's mono_inertial_video) and <prefix>.csv (IMU, matching
  // robo_manip_baselines/misc/ConvertInsvToMp4AndImu.py's long format:
  // header "timestamp_ms,sensor_type,x,y,z", one gyro row then one accel
  // row per sample sharing the same timestamp -- mono_inertial_video.cc's
  // LoadIMUCsv requires exactly that pairing/order).
  cv::VideoWriter record_writer;
  std::ofstream record_csv;
  bool record_csv_header_written = false;
  // frame_timestamp_sec (the real camera-SDK timestamp, same clock domain
  // as imu_meas's timestamps) used to be computed per frame but never
  // saved anywhere -- only used live for broadcast/SLAM tracking, leaving
  // consumers of a --record'ed video (e.g. an offline camera-IMU
  // calibration, or mono_inertial_video.cc) to assume a constant frame
  // rate instead of the real (slightly jittery) capture times. A sidecar
  // CSV alongside the video/IMU CSV fixes that.
  std::ofstream record_frame_ts_csv;
  bool record_frame_ts_header_written = false;
  int record_frame_index = 0;

  BridgeServer server(socket_path);
  server.Start();
  std::cout << "[insta360_bridge] Serving on " << socket_path << std::endl;
  if (!record_prefix.empty()) {
    std::cout << "[insta360_bridge] Recording to " << record_prefix
              << ".avi / .csv" << std::endl;
  }
  if (use_preview) {
    cv::namedWindow("insta360_bridge preview", cv::WINDOW_NORMAL);
    cv::resizeWindow("insta360_bridge preview", kSlamFrameSize,
                      kSlamFrameSize);
  }

  // Dedicated SLAM thread: wakes on every new decoded frame
  // (WaitForNextFrame, ~30fps) so no frame is missed, but only actually
  // calls TrackMonocular for a frame once real (non-synthetic) IMU
  // coverage is confirmed to reach at least that frame's own timestamp
  // (GetImuWatermarkSec) -- see that accessor's comment. This replaced two
  // earlier designs: gyro-batch-driven triggering (guaranteed non-empty
  // windows but only processed a frame every ~100ms, worsening "Fail to
  // track local map!" resets from the larger inter-frame visual baseline)
  // and immediate frame-driven triggering with possibly-empty IMU windows
  // (processed every frame instantly, but left many frames -- and any
  // keyframe born from one -- permanently without preintegrated IMU data,
  // eventually starving LocalMapping's inertial bundle adjustment of edges
  // and crashing with "Sophus SO3::exp failed! omega: nan nan nan"; a
  // zero-order-hold synthetic-sample workaround was also tried and
  // reverted, see git history, since it could violate OnGyroData's batch
  // timestamp ordering). Buffering frames and submitting them once the
  // watermark catches up keeps every frame processed (no visual gap
  // reintroduced) while guaranteeing each one gets a real, correctly-timed
  // IMU window -- typically only 0-1 batch-interval (~100ms) of lag, not
  // an architectural revert. Not started for --no-slam or when the
  // accel-carrying gyro path isn't compiled in.
  std::thread slam_thread;
#if INSTA360_GYRO_HAS_ACCEL
  if (!no_slam) {
    slam_thread = std::thread([&]() {
      cv::Mat thread_safe_mask;
      std::vector<ORB_SLAM3::IMU::Point> thread_imu_meas;
      uint64_t last_seen_seq = 0;
      // Bounded so a prolonged gyro outage (SDK stops delivering batches)
      // can't grow this without limit -- drop the oldest pending frame
      // rather than lag forever. ~15 frames is ~0.5s at 30fps, several
      // times the normal ~1 gyro-batch-interval (~100-170ms) lag.
      constexpr size_t kMaxPendingFrames = 15;
      std::deque<std::pair<cv::Mat, double>> pending_frames;
      while (g_running) {
        uint64_t seq = delegate->WaitForNextFrame(last_seen_seq);
        if (seq == last_seen_seq) {
          // WaitForNextFrame only returns without advancing seq when
          // g_running went false while it was waiting.
          break;
        }
        last_seen_seq = seq;

        // --no-imu: plain MONOCULAR tracking, so none of the
        // IMU-warmup-gate/watermark-buffering machinery below applies --
        // just process every frame immediately with an empty IMU vector
        // (System::TrackMonocular ignores it entirely when mSensor isn't
        // IMU_MONOCULAR, see System.cc).
        if (no_imu) {
          cv::Mat frame;
          double frame_timestamp_sec;
          if (!delegate->PeekLatestFrame(frame, frame_timestamp_sec)) {
            continue;
          }
          cv::Mat slam_frame =
              BuildSlamFrame(frame, use_back_lens, thread_safe_mask);
          Sophus::SE3f Tcw =
              slam->TrackMonocular(slam_frame, frame_timestamp_sec, {});
          Sophus::SE3f Twc = Tcw.inverse();
          Eigen::Vector3f t = Twc.translation();
          Eigen::Quaternionf q = Twc.unit_quaternion();
          std::string tracking_state =
              TrackingStateToString(slam->GetTrackingState());
          server.Broadcast(EncodePoseMessage(
              frame_timestamp_sec, t.x(), t.y(), t.z(), q.w(), q.x(), q.y(),
              q.z(), tracking_state));
          continue;
        }

        if (!delegate->HasReceivedAnyGyroData()) {
          // See HasReceivedAnyGyroData's comment -- nothing for
          // ORB_SLAM3's internal IMU queue to draw on yet at all.
          continue;
        }

        cv::Mat frame;
        double frame_timestamp_sec;
        if (delegate->PeekLatestFrame(frame, frame_timestamp_sec) &&
            (pending_frames.empty() ||
             frame_timestamp_sec > pending_frames.back().second)) {
          pending_frames.emplace_back(frame, frame_timestamp_sec);
          while (pending_frames.size() > kMaxPendingFrames) {
            std::cout << "[insta360_bridge] SLAM thread dropping oldest "
                         "pending frame -- IMU watermark not keeping up "
                         "(gyro delivery stalled?)"
                      << std::endl;
            pending_frames.pop_front();
          }
        }

        const double watermark_sec = delegate->GetImuWatermarkSec();
        while (!pending_frames.empty() &&
               pending_frames.front().second <= watermark_sec) {
          cv::Mat ready_frame = pending_frames.front().first;
          double ready_timestamp_sec = pending_frames.front().second;
          pending_frames.pop_front();

          for (const auto& sample : delegate->TakeImuSamples()) {
            thread_imu_meas.emplace_back(
                sample.accel[0], sample.accel[1], sample.accel[2],
                sample.gyro[0], sample.gyro[1], sample.gyro[2],
                sample.timestamp_sec);
          }

          cv::Mat slam_frame =
              BuildSlamFrame(ready_frame, use_back_lens, thread_safe_mask);

          Sophus::SE3f Tcw = slam->TrackMonocular(
              slam_frame, ready_timestamp_sec, thread_imu_meas);
          thread_imu_meas.clear();

          Sophus::SE3f Twc = Tcw.inverse();
          Eigen::Vector3f t = Twc.translation();
          Eigen::Quaternionf q = Twc.unit_quaternion();
          std::string tracking_state =
              TrackingStateToString(slam->GetTrackingState());
          server.Broadcast(EncodePoseMessage(
              ready_timestamp_sec, t.x(), t.y(), t.z(), q.w(), q.x(), q.y(),
              q.z(), tracking_state));
        }
      }
    });
  }
#endif

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

    for (const auto& sample : delegate->TakeImuSamplesForRecording()) {
      imu_meas.emplace_back(sample.accel[0], sample.accel[1], sample.accel[2],
                             sample.gyro[0], sample.gyro[1], sample.gyro[2],
                             sample.timestamp_sec);
    }

    if (!need_slam_frame) {
      imu_meas.clear();
      continue;
    }

    // See BuildSlamFrame's comment for the full crop/mask rationale. The
    // full dual-fisheye frame above is still broadcast as-is for any other
    // consumer; this crop is only used for recording/preview/SLAM. Used
    // here (not just by the SLAM thread) so a --no-slam recording matches
    // the same calibration/mask.
    cv::Mat slam_frame = BuildSlamFrame(frame, use_back_lens, safe_mask);

    if (!record_prefix.empty()) {
      if (!record_writer.isOpened()) {
        // MJPG in an AVI container (not mp4v in .mp4): mp4v is inter-frame
        // (motion-compensated) and this OpenCV/ffmpeg build gives it no
        // usable bitrate/quality control (VIDEOWRITER_PROP_QUALITY silently
        // ignored, confirmed empirically) -- its hardcoded default bitrate
        // is too low once motion vectors can't track fast panning,
        // producing macroblock corruption on exactly the frames where the
        // camera is moving (confirmed via frame-by-frame inspection of a
        // real recording: clean at rest, blocky ceiling/wall artifacts
        // during pans). MJPG encodes every frame independently (no motion
        // prediction), so there's nothing for fast motion to break; each
        // frame's quality is constant regardless of motion. The .mp4/mov
        // muxer rejects the MJPEG codec tag outright (confirmed: ffmpeg
        // silently falls back to mp4v, defeating the fix) -- AVI accepts
        // it natively, and is still readable by cv::VideoCapture (for
        // mono_inertial_video) via the same ffmpeg backend. 30fps matches
        // the camera's configured rate (--no-slam mode's throughput is
        // close to but not exactly 30fps -- accepted as an approximation,
        // see README).
        record_writer.open(record_prefix + ".avi",
                            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0,
                            cv::Size(kSlamFrameSize, kSlamFrameSize));
        if (!record_writer.isOpened()) {
          std::cerr << "[insta360_bridge] Failed to open " << record_prefix
                     << ".avi for recording." << std::endl;
        }
        record_csv.open(record_prefix + ".csv");
        // Default ostream precision (6 significant digits) truncates a
        // ~1000+ms timestamp down to ~2 decimal places -- confirmed this
        // silently destroyed the sub-millisecond spacing OnGyroData's
        // interpolation (above) computes to fix the SDK's coarse/batched
        // raw timestamps, making genuinely-distinct in-memory values
        // collapse back into visible duplicates once written to disk.
        record_csv << std::fixed << std::setprecision(6);
        record_frame_ts_csv.open(record_prefix + "_frame_timestamps.csv");
        record_frame_ts_csv << std::fixed << std::setprecision(6);
      }
      if (record_writer.isOpened()) {
        record_writer.write(slam_frame);
        if (record_frame_ts_csv.is_open()) {
          if (!record_frame_ts_header_written) {
            record_frame_ts_csv << "frame_index,timestamp_ms\n";
            record_frame_ts_header_written = true;
          }
          record_frame_ts_csv << record_frame_index << ","
                               << frame_timestamp_sec * 1000.0 << "\n";
          record_frame_index++;
        }
      }
      if (record_csv.is_open()) {
        if (!record_csv_header_written) {
          record_csv << "timestamp_ms,sensor_type,x,y,z\n";
          record_csv_header_written = true;
        }
        // Matches ConvertInsvToMp4AndImu.py's long format: one gyro row
        // then one accel row per sample, sharing the same timestamp_ms --
        // mono_inertial_video.cc's LoadIMUCsv requires exactly this
        // pairing/order.
        for (const auto& m : imu_meas) {
          double t_ms = m.t * 1000.0;
          record_csv << t_ms << ",gyro," << m.w.x() << "," << m.w.y() << ","
                     << m.w.z() << "\n";
          record_csv << t_ms << ",accel," << m.a.x() << "," << m.a.y() << ","
                     << m.a.z() << "\n";
        }
      }
    }

    if (use_preview) {
      cv::imshow("insta360_bridge preview", slam_frame);
      cv::waitKey(1);
    }

    // Actual SLAM processing (TrackMonocular + pose broadcast) now happens
    // on a dedicated thread, triggered directly by new gyro bursts rather
    // than by this loop's frame polling -- see the slam_thread lambda
    // below. This loop's own imu_meas was only needed for the --record
    // CSV above.
    imu_meas.clear();
  }
  if (slam_thread.joinable()) {
    // g_running is already false here (that's what ended the loop above),
    // so WaitForNextFrame will notice on its next 200ms wait_for timeout
    // at the latest and the thread will exit on its own -- this just
    // waits for that to actually happen before touching slam/camera
    // below.
    slam_thread.join();
  }
  if (record_writer.isOpened()) {
    record_writer.release();
  }
  if (record_csv.is_open()) {
    record_csv.close();
  }
  if (record_frame_ts_csv.is_open()) {
    record_frame_ts_csv.close();
  }
  if (use_preview) {
    cv::destroyAllWindows();
  }

  std::cout << "[insta360_bridge] Shutting down." << std::endl;
  camera->StopLiveStreaming();
  camera->Close();
  server.Stop();
  if (no_slam) {
    return 0;
  }
  if (use_viewer) {
    // Known upstream ORB_SLAM3 limitation, not specific to this bridge:
    // System::Shutdown() (third_party/ORB_SLAM3/src/System.cc) has its
    // mpViewer->RequestFinish()/isFinished() wait commented out, so it
    // never synchronizes with the still-running Pangolin viewer thread
    // (mptViewer) before this process's objects start being destructed --
    // verified against real hardware that this races and crashes
    // (SIGSEGV/core dump) on exit whenever --viewer was used. mpViewer/
    // mptViewer are private to System, so there's nothing to join from
    // here either. _exit() skips the C++ destructor chain entirely,
    // sidestepping the race -- acceptable since --viewer is an interactive
    // debug aid (not production teleop, which runs headless and never
    // creates a Viewer thread in the first place).
    std::_Exit(0);
  }
  slam->Shutdown();
  return 0;
}
