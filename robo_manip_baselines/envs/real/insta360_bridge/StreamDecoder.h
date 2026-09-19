// Decodes the H.264/H.265 preview-stream bytes delivered by
// ins_camera::StreamDelegate::OnVideoData (per Desktop-CameraSDK-Cpp's
// README, the preview stream is NOT raw frames) into cv::Mat BGR frames,
// using ffmpeg/libavcodec. Standard streaming-decode pattern: feed
// arbitrarily-chunked compressed bytes in via Decode(), get a frame out
// whenever the decoder has accumulated a full one (which is not
// necessarily on every call).
#pragma once

#include <cstdint>

#include <opencv2/opencv.hpp>

extern "C" {
struct AVCodecContext;
struct AVCodecParserContext;
struct SwsContext;
struct AVFrame;
struct AVPacket;
}

class StreamDecoder {
 public:
  // is_h265: from camera->GetVideoEncodeType() (see main.cc) -- the SDK
  // README states the encoding is either H.264 or H.265 depending on the
  // camera/mode, queried once after StartLiveStreaming.
  explicit StreamDecoder(bool is_h265);
  ~StreamDecoder();

  StreamDecoder(const StreamDecoder&) = delete;
  StreamDecoder& operator=(const StreamDecoder&) = delete;

  // Feeds one chunk of compressed video bytes (as received verbatim from
  // OnVideoData). Returns true and fills out_frame (BGR, HxWx3 uint8) once
  // per fully decoded frame; may return false (and leave out_frame
  // untouched) for a given call if the decoder needs more data first.
  bool Decode(const uint8_t* data, size_t size, cv::Mat& out_frame);

 private:
  AVCodecContext* codec_ctx_ = nullptr;
  AVCodecParserContext* parser_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  AVFrame* av_frame_ = nullptr;
  AVFrame* bgr_frame_ = nullptr;
  AVPacket* packet_ = nullptr;
};
