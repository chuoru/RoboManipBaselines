#include "BridgeServer.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

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
    std::lock_guard<std::mutex> lock(clients_mutex_);
    client_fds_.push_back(client_fd);
    std::printf("[BridgeServer] Client connected (fd=%d).\n", client_fd);
  }
}

void BridgeServer::Broadcast(const std::vector<uint8_t>& message) {
  std::lock_guard<std::mutex> lock(clients_mutex_);
  std::vector<int> remaining;
  remaining.reserve(client_fds_.size());
  for (int fd : client_fds_) {
    // MSG_NOSIGNAL: a client that has disconnected must not raise SIGPIPE
    // and take the whole bridge process down with it.
    ssize_t sent = ::send(fd, message.data(), message.size(), MSG_NOSIGNAL);
    if (sent == static_cast<ssize_t>(message.size())) {
      remaining.push_back(fd);
    } else {
      ::close(fd);
    }
  }
  client_fds_ = std::move(remaining);
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

  std::lock_guard<std::mutex> lock(clients_mutex_);
  for (int fd : client_fds_) {
    ::close(fd);
  }
  client_fds_.clear();

  ::unlink(socket_path_.c_str());
}
