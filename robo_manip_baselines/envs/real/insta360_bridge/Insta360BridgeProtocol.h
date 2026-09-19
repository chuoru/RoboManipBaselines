// Wire protocol shared with the Python side -- MUST stay in lockstep with
// common/utils/Insta360Protocol.py. See that file's module docstring for the
// full framing spec:
//   4 bytes            big-endian uint32 header_len
//   header_len bytes   UTF-8 JSON header
//   (if header["type"] == "frame")
//       w * h * 3 bytes   raw RGB, row-major, uint8
#pragma once

#include <arpa/inet.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Encodes one "frame" message: header + raw RGB payload.
inline std::vector<uint8_t> EncodeFrameMessage(double timestamp, int width,
                                                int height,
                                                const uint8_t* rgb_data) {
  char header_buf[128];
  int header_len = std::snprintf(
      header_buf, sizeof(header_buf),
      R"({"type":"frame","t":%.6f,"w":%d,"h":%d})", timestamp, width, height);

  std::vector<uint8_t> message(4 + header_len + static_cast<size_t>(width) *
                                                      height * 3);
  uint32_t header_len_be = htonl(static_cast<uint32_t>(header_len));
  std::memcpy(message.data(), &header_len_be, 4);
  std::memcpy(message.data() + 4, header_buf, header_len);
  std::memcpy(message.data() + 4 + header_len, rgb_data,
              static_cast<size_t>(width) * height * 3);
  return message;
}

// Encodes one "pose" message. tracking_state is one of "OK"/"LOST"/"INIT".
inline std::vector<uint8_t> EncodePoseMessage(
    double timestamp, float px, float py, float pz, float qw, float qx,
    float qy, float qz, const std::string& tracking_state) {
  char header_buf[256];
  int header_len = std::snprintf(
      header_buf, sizeof(header_buf),
      R"({"type":"pose","t":%.6f,"pos":[%.6f,%.6f,%.6f],)"
      R"("quat":[%.6f,%.6f,%.6f,%.6f],"tracking_state":"%s"})",
      timestamp, px, py, pz, qw, qx, qy, qz, tracking_state.c_str());

  std::vector<uint8_t> message(4 + header_len);
  uint32_t header_len_be = htonl(static_cast<uint32_t>(header_len));
  std::memcpy(message.data(), &header_len_be, 4);
  std::memcpy(message.data() + 4, header_buf, header_len);
  return message;
}
