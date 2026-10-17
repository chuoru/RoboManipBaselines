#include "BridgeServer.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {

// Mirrors common/utils/Insta360Protocol.py's recv_exact: reads exactly
// num_bytes from fd, or returns false on a clean EOF/error (never partial).
bool RecvExact(int fd, void* buf, size_t num_bytes) {
  size_t received = 0;
  uint8_t* p = static_cast<uint8_t*>(buf);
  while (received < num_bytes) {
    ssize_t n = ::recv(fd, p + received, num_bytes - received, 0);
    if (n <= 0) {
      return false;
    }
    received += static_cast<size_t>(n);
  }
  return true;
}

}  // namespace

BridgeServer::BridgeServer(const std::string& socket_path)
    : socket_path_(socket_path) {
  ::unlink(socket_path_.c_str());

  listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    throw std::runtime_error("BridgeServer: socket() failed");
  }

  struct sockaddr_un addr {};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  if (::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr),
             sizeof(addr)) < 0) {
    throw std::runtime_error("BridgeServer: bind() failed for " +
                              socket_path_);
  }
  if (::listen(listen_fd_, 4) < 0) {
    throw std::runtime_error("BridgeServer: listen() failed");
  }
}

BridgeServer::~BridgeServer() { Stop(); }

void BridgeServer::SetMessageCallback(
    std::function<void(const std::string&)> callback) {
  message_callback_ = std::move(callback);
}

void BridgeServer::Start() {
  running_ = true;
  accept_thread_ = std::thread(&BridgeServer::AcceptLoop, this);
}

void BridgeServer::AcceptLoop() {
  while (running_) {
    int client_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (running_) {
        std::perror("BridgeServer: accept() failed");
      }
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(clients_mutex_);
      client_fds_.push_back(client_fd);
      reader_threads_.emplace_back(&BridgeServer::ClientReadLoop, this,
                                    client_fd);
    }
    std::printf("[BridgeServer] Client connected (fd=%d).\n", client_fd);
  }
}

void BridgeServer::ClientReadLoop(int client_fd) {
  // Same length-prefixed-JSON framing as Broadcast()'s messages (see
  // Insta360BridgeProtocol.h), but incoming client->server messages are
  // always header-only -- no "frame"-style binary payload to read.
  while (running_) {
    uint32_t header_len_be = 0;
    if (!RecvExact(client_fd, &header_len_be, sizeof(header_len_be))) {
      break;
    }
    uint32_t header_len = ntohl(header_len_be);
    // Control messages are tiny (a type string and maybe a short path) --
    // reject anything implausibly large rather than attempting a huge
    // allocation on a corrupt/malicious length field.
    if (header_len > 1 << 16) {
      std::fprintf(stderr,
                    "[BridgeServer] Client fd=%d sent implausible header "
                    "length %u, closing connection.\n",
                    client_fd, header_len);
      break;
    }
    std::string header(header_len, '\0');
    // &header[0] rather than .data(): this project targets C++14, where
    // std::string::data() is const-only (the non-const overload is a
    // C++17 addition).
    if (header_len > 0 && !RecvExact(client_fd, &header[0], header_len)) {
      break;
    }
    if (message_callback_) {
      message_callback_(header);
    }
  }
  RemoveClient(client_fd);
}

void BridgeServer::RemoveClient(int client_fd) {
  std::lock_guard<std::mutex> lock(clients_mutex_);
  auto it = std::find(client_fds_.begin(), client_fds_.end(), client_fd);
  if (it != client_fds_.end()) {
    client_fds_.erase(it);
    ::close(client_fd);
  }
}

void BridgeServer::Broadcast(const std::vector<uint8_t>& message) {
  std::lock_guard<std::mutex> lock(clients_mutex_);
  for (int fd : client_fds_) {
    // MSG_NOSIGNAL: a client that has disconnected must not raise SIGPIPE
    // and take the whole bridge process down with it.
    ssize_t sent = ::send(fd, message.data(), message.size(), MSG_NOSIGNAL);
    if (sent != static_cast<ssize_t>(message.size())) {
      // Don't close()/erase here -- RemoveClient (called from this fd's
      // ClientReadLoop thread) is the single owner of client_fds_
      // erasure/closing, so two code paths can't race to close() the
      // same fd number (which the OS could have already reused for an
      // unrelated socket by the time a second close() ran).
      // shutdown() makes that thread's blocking recv() return promptly
      // so it notices and cleans up.
      ::shutdown(fd, SHUT_RDWR);
    }
  }
}

void BridgeServer::Stop() {
  if (!running_) {
    return;
  }
  running_ = false;

  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }

  // Shut down (not close) every client fd so each ClientReadLoop's
  // blocking recv() returns promptly; that loop's own RemoveClient() call
  // does the actual close()/erase (see Broadcast()'s comment for why this
  // function must not also close them).
  std::vector<std::thread> threads_to_join;
  {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (int fd : client_fds_) {
      ::shutdown(fd, SHUT_RDWR);
    }
    threads_to_join = std::move(reader_threads_);
  }
  for (auto& t : threads_to_join) {
    if (t.joinable()) {
      t.join();
    }
  }

  ::unlink(socket_path_.c_str());
}
