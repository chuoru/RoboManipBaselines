// Unix domain socket server: accepts any number of client connections (in
// practice two -- RealEnvBase.setup_insta360 for "frame" messages and
// Insta360InputDevice for "pose" messages, see Insta360BridgeProtocol.h) and
// broadcasts every message sent via Broadcast() to all of them.
#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <vector>

class BridgeServer {
 public:
  // Removes any stale socket file at socket_path before binding.
  explicit BridgeServer(const std::string& socket_path);
  ~BridgeServer();

  // Starts the background accept-loop thread. Call once before Broadcast().
  void Start();

  // Sends `message` to every currently connected client; silently drops any
  // client whose send fails (it will simply stop receiving further
  // messages -- the Python side treats that as a stale/disconnected bridge).
  void Broadcast(const std::vector<uint8_t>& message);

  void Stop();

 private:
  void AcceptLoop();

  std::string socket_path_;
  int listen_fd_ = -1;
  std::thread accept_thread_;
  bool running_ = false;

  std::mutex clients_mutex_;
  std::vector<int> client_fds_;
};
