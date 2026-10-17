// Wire protocol shared with the Python side -- MUST stay in lockstep with
// common/utils/Insta360Protocol.py. See that file's module docstring for the
// full framing spec:
//   4 bytes            big-endian uint32 header_len
//   header_len bytes   UTF-8 JSON header
//   (if header["type"] == "frame")
//       w * h * 3 bytes   raw RGB, row-major, uint8
//
// Client -> server control messages (handled in main.cc, received via
// BridgeServer::SetMessageCallback) are header-only, same framing, one of:
//   {"type":"start_recording"}
//   {"type":"stop_recording"}
// and get a reply broadcast back (so it reaches whichever client sent the
// request, since BridgeServer::Broadcast has no per-client targeting):
//   {"type":"recording_started","ok":true|false,"error":"..." (if !ok)}
//   {"type":"recording_stopped","ok":true|false,"path":"..." (if ok),
//    "error":"..." (if !ok)}
// "path" is a local filesystem path to the downloaded .insv file (see
// main.cc's recording-control callback), ready for
// misc/ConvertInsvToMp4AndImu.py.
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

// Helper shared by the two encoders below.
inline std::vector<uint8_t> EncodeHeaderOnlyMessage(const std::string& json) {
  uint32_t header_len_be = htonl(static_cast<uint32_t>(json.size()));
  std::vector<uint8_t> message(4 + json.size());
  std::memcpy(message.data(), &header_len_be, 4);
  std::memcpy(message.data() + 4, json.data(), json.size());
  return message;
}

inline std::vector<uint8_t> EncodeRecordingStartedMessage(
    bool ok, const std::string& error = "") {
  char buf[256];
  int len = ok ? std::snprintf(buf, sizeof(buf),
                                R"({"type":"recording_started","ok":true})")
               : std::snprintf(
                     buf, sizeof(buf),
                     R"({"type":"recording_started","ok":false,"error":"%s"})",
                     error.c_str());
  return EncodeHeaderOnlyMessage(std::string(buf, len));
}

inline std::vector<uint8_t> EncodeRecordingStoppedMessage(
    bool ok, const std::string& path_or_error) {
  char buf[512];
  int len =
      ok ? std::snprintf(buf, sizeof(buf),
                          R"({"type":"recording_stopped","ok":true,"path":"%s"})",
                          path_or_error.c_str())
         : std::snprintf(
               buf, sizeof(buf),
               R"({"type":"recording_stopped","ok":false,"error":"%s"})",
               path_or_error.c_str());
  return EncodeHeaderOnlyMessage(std::string(buf, len));
}

// Extracts the top-level "type" string field from a small, flat JSON
// object (e.g. {"type":"start_recording"}). Hand-rolled rather than
// pulling in a JSON library, since every message this bridge needs to
// DECODE (as opposed to encode, above) is this simple by construction --
// see common/utils/Insta360Protocol.py's encode_message, the only sender.
// Returns false if no "type" field is found.
inline bool ExtractMessageType(const std::string& json, std::string* out) {
  const std::string key = "\"type\"";
  size_t key_pos = json.find(key);
  if (key_pos == std::string::npos) {
    return false;
  }
  size_t colon_pos = json.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) {
    return false;
  }
  size_t value_start = json.find('"', colon_pos + 1);
  if (value_start == std::string::npos) {
    return false;
  }
  size_t value_end = json.find('"', value_start + 1);
  if (value_end == std::string::npos) {
    return false;
  }
  *out = json.substr(value_start + 1, value_end - value_start - 1);
  return true;
}
