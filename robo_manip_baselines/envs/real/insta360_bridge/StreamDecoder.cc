#include "StreamDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#include <stdexcept>

StreamDecoder::StreamDecoder(bool is_h265) {
  const AVCodec* codec = avcodec_find_decoder(
      is_h265 ? AV_CODEC_ID_HEVC : AV_CODEC_ID_H264);
  if (!codec) {
    throw std::runtime_error("StreamDecoder: decoder not found");
  }

  codec_ctx_ = avcodec_alloc_context3(codec);
  if (!codec_ctx_ || avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
    throw std::runtime_error("StreamDecoder: avcodec_open2 failed");
  }

  parser_ = av_parser_init(codec->id);
  if (!parser_) {
    throw std::runtime_error("StreamDecoder: av_parser_init failed");
  }

  av_frame_ = av_frame_alloc();
  bgr_frame_ = av_frame_alloc();
  packet_ = av_packet_alloc();
}

StreamDecoder::~StreamDecoder() {
  if (sws_ctx_) sws_freeContext(sws_ctx_);
  if (bgr_frame_) av_frame_free(&bgr_frame_);
  if (av_frame_) av_frame_free(&av_frame_);
  if (packet_) av_packet_free(&packet_);
  if (parser_) av_parser_close(parser_);
  if (codec_ctx_) avcodec_free_context(&codec_ctx_);
}

bool StreamDecoder::Decode(const uint8_t* data, size_t size,
                            cv::Mat& out_frame) {
  const uint8_t* cursor = data;
  size_t remaining = size;
  bool got_frame = false;

  // av_parser_parse2 may need several calls to accumulate one complete
  // Annex-B packet, especially for the first NAL units of a chunk that
  // itself spans multiple encoded frames.
  while (remaining > 0) {
    uint8_t* parsed_data = nullptr;
    int parsed_size = 0;
    int consumed = av_parser_parse2(
        parser_, codec_ctx_, &parsed_data, &parsed_size, cursor,
        static_cast<int>(remaining), AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
    if (consumed < 0) {
      break;
    }
    cursor += consumed;
    remaining -= consumed;

    if (parsed_size == 0) {
      continue;
    }

    packet_->data = parsed_data;
    packet_->size = parsed_size;
    if (avcodec_send_packet(codec_ctx_, packet_) < 0) {
      continue;
    }

    while (avcodec_receive_frame(codec_ctx_, av_frame_) == 0) {
      if (!sws_ctx_) {
        sws_ctx_ = sws_getContext(
            av_frame_->width, av_frame_->height,
            static_cast<AVPixelFormat>(av_frame_->format), av_frame_->width,
            av_frame_->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr,
            nullptr, nullptr);
        out_frame.create(av_frame_->height, av_frame_->width, CV_8UC3);
      }

      uint8_t* dst_data[4] = {out_frame.data, nullptr, nullptr, nullptr};
      int dst_linesize[4] = {static_cast<int>(out_frame.step), 0, 0, 0};
      sws_scale(sws_ctx_, av_frame_->data, av_frame_->linesize, 0,
                av_frame_->height, dst_data, dst_linesize);
      got_frame = true;
    }
  }

  return got_frame;
}
