// Unix domain socket server: accepts any number of client connections (in
// practice several -- RealEnvBase.setup_insta360 for "frame" messages,
// Insta360InputDevice for "pose" messages, and a short-lived connection per
// episode to send "start_recording"/"stop_recording" control messages, see
// Insta360BridgeProtocol.h) and broadcasts every message sent via
// Broadcast() to all of them.
//
// Client -> server messages (received via the per-client reader thread
// started in AcceptLoop, and handed to the callback set with
// SetMessageCallback) use the SAME length-prefixed-JSON framing as
// server -> client messages, but are always header-only (no binary
// payload) -- see Insta360BridgeProtocol.h's ExtractJsonStringField for
// how main.cc parses them.
#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class BridgeServer {
 public:
  // Removes any stale socket file at socket_path before binding.
  explicit BridgeServer(const std::string& socket_path);
  ~BridgeServer();

  // Called (from a per-client reader thread -- NOT the thread that calls
  // Start()) with the raw JSON header string of every header-only message
  // received from any client. Must be set before Start() to avoid a race
  // with an already-connected client's reader thread. Not reentrant-safe
  // to call concurrently from multiple clients' reader threads unless the
  // callback itself is thread-safe.
  void SetMessageCallback(std::function<void(const std::string&)> callback);

  // Starts the background accept-loop thread. Call once before Broadcast().
  void Start();

  // Sends `message` to every currently connected client; silently drops any
  // client whose send fails (it will simply stop receiving further
  // messages -- the Python side treats that as a stale/disconnected bridge).
  void Broadcast(const std::vector<uint8_t>& message);

  void Stop();

 private:
  void AcceptLoop();
  void ClientReadLoop(int client_fd);
  void RemoveClient(int client_fd);

  std::string socket_path_;
  int listen_fd_ = -1;
  std::thread accept_thread_;
  bool running_ = false;

  std::mutex clients_mutex_;
  std::vector<int> client_fds_;
  std::vector<std::thread> reader_threads_;

  std::function<void(const std::string&)> message_callback_;
};
