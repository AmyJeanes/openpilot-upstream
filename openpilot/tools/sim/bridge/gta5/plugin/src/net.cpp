#include <winsock2.h>
#include <ws2tcpip.h>

#include "net.h"

#include <cstdlib>
#include <cstring>
#include <functional>

#pragma comment(lib, "ws2_32.lib")

namespace {

void SkipSpace(const std::string &s, size_t &i) {
  while (i < s.size() && isspace(static_cast<unsigned char>(s[i]))) i++;
}

bool ParseString(const std::string &s, size_t &i, std::string &out) {
  if (i >= s.size() || s[i] != '"') return false;
  out.clear();
  for (i++; i < s.size(); i++) {
    char c = s[i];
    if (c == '"') {
      i++;
      return true;
    }
    if (c == '\\' && i + 1 < s.size()) {
      char e = s[++i];
      out += e == 'n' ? '\n' : e == 't' ? '\t' : e;
    } else {
      out += c;
    }
  }
  return false;
}

}  // namespace

bool ParseFlatJson(const std::string &s, Message &out) {
  size_t i = 0;
  SkipSpace(s, i);
  if (i >= s.size() || s[i++] != '{') return false;
  for (;;) {
    SkipSpace(s, i);
    if (i < s.size() && s[i] == '}') return true;
    std::string key, value;
    if (!ParseString(s, i, key)) return false;
    SkipSpace(s, i);
    if (i >= s.size() || s[i++] != ':') return false;
    SkipSpace(s, i);
    if (i < s.size() && s[i] == '"') {
      if (!ParseString(s, i, value)) return false;
    } else {
      size_t start = i;
      while (i < s.size() && s[i] != ',' && s[i] != '}') i++;
      value = s.substr(start, i - start);
      while (!value.empty() && isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    }
    out[key] = value;
    SkipSpace(s, i);
    if (i < s.size() && s[i] == ',') i++;
  }
}

double MsgNum(const Message &m, const char *key, double fallback) {
  auto it = m.find(key);
  if (it == m.end()) return fallback;
  char *end = nullptr;
  double v = strtod(it->second.c_str(), &end);
  return end == it->second.c_str() ? fallback : v;
}

bool MsgBool(const Message &m, const char *key, bool fallback) {
  auto it = m.find(key);
  if (it == m.end()) return fallback;
  return it->second == "true" || std::atof(it->second.c_str()) != 0;  // gta5_cmd.py sends on=1 as 1.0
}

std::string MsgStr(const Message &m, const char *key, const std::string &fallback) {
  auto it = m.find(key);
  return it == m.end() ? fallback : it->second;
}

void Link::Start(std::function<std::string()> address, std::function<void(const std::string &)> log) {
  address_ = std::move(address);
  log_ = std::move(log);
  stop_ = false;
  thread_ = std::thread([this] { Run(); });
}

void Link::Stop() {
  stop_ = true;
  uintptr_t s = sock_.exchange(INVALID_SOCKET);
  if (s != INVALID_SOCKET) {
    shutdown(static_cast<SOCKET>(s), SD_BOTH);
    closesocket(static_cast<SOCKET>(s));
  }
  sendCv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Link::SendFrame(const std::string &header, std::vector<uint8_t> &&payload) {
  if (!connected_) return;
  uint32_t headLen = static_cast<uint32_t>(header.size());
  uint32_t total = 4 + headLen + static_cast<uint32_t>(payload.size());
  std::vector<uint8_t> msg(4 + total);
  memcpy(msg.data(), &total, 4);
  memcpy(msg.data() + 4, &headLen, 4);
  memcpy(msg.data() + 8, header.data(), headLen);
  memcpy(msg.data() + 8 + headLen, payload.data(), payload.size());
  {
    std::lock_guard lk(sendMutex_);
    pending_ = std::move(msg);
    hasPending_ = true;
  }
  sendCv_.notify_one();
}

std::vector<Message> Link::TakeMessages() {
  std::lock_guard lk(recvMutex_);
  std::vector<Message> out(received_.begin(), received_.end());
  received_.clear();
  return out;
}

void Link::Receive(uintptr_t sockv) {
  SOCKET sock = static_cast<SOCKET>(sockv);
  std::string buf;
  char chunk[4096];
  for (;;) {
    int n = recv(sock, chunk, sizeof(chunk), 0);
    if (n <= 0) return;
    buf.append(chunk, n);
    size_t nl;
    while ((nl = buf.find('\n')) != std::string::npos) {
      Message m;
      if (ParseFlatJson(buf.substr(0, nl), m)) {
        std::lock_guard lk(recvMutex_);
        received_.push_back(std::move(m));
        if (received_.size() > 1000) received_.pop_front();
      }
      buf.erase(0, nl + 1);
    }
  }
}

void Link::Run() {
  WSADATA wsa;
  WSAStartup(MAKEWORD(2, 2), &wsa);
  std::string lastError;
  while (!stop_) {
    std::string addr = address_();
    size_t colon = addr.rfind(':');
    SOCKET sock = INVALID_SOCKET;
    if (colon != std::string::npos) {
      addrinfo hints{}, *res = nullptr;
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_STREAM;
      if (getaddrinfo(addr.substr(0, colon).c_str(), addr.substr(colon + 1).c_str(), &hints, &res) == 0) {
        sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        // a short connect timeout: the bridge's address goes stale when WSL restarts
        u_long nb = 1;
        ioctlsocket(sock, FIONBIO, &nb);
        connect(sock, res->ai_addr, static_cast<int>(res->ai_addrlen));
        fd_set w;
        FD_ZERO(&w);
        FD_SET(sock, &w);
        timeval tv{1, 0};
        int err = 0, len = sizeof(err);
        if (select(0, nullptr, &w, nullptr, &tv) != 1 ||
            getsockopt(sock, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&err), &len) != 0 || err != 0) {
          closesocket(sock);
          sock = INVALID_SOCKET;
        } else {
          nb = 0;
          ioctlsocket(sock, FIONBIO, &nb);
        }
        freeaddrinfo(res);
      }
    }
    if (sock == INVALID_SOCKET) {
      for (int i = 0; i < 10 && !stop_; i++) Sleep(100);
      continue;
    }
    BOOL nodelay = TRUE;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char *>(&nodelay), sizeof(nodelay));
    int sndbuf = 16 << 20;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<char *>(&sndbuf), sizeof(sndbuf));
    sock_ = sock;
    connected_ = true;
    log_("connected to the bridge at " + addr);

    std::thread rx([this, sock] {
      Receive(sock);
      connected_ = false;
      sendCv_.notify_all();
    });
    while (!stop_ && connected_) {
      std::vector<uint8_t> msg;
      {
        std::unique_lock lk(sendMutex_);
        sendCv_.wait(lk, [this] { return hasPending_ || stop_ || !connected_; });
        if (!hasPending_) continue;
        msg = std::move(pending_);
        hasPending_ = false;
      }
      size_t off = 0;
      while (off < msg.size()) {
        int n = send(sock, reinterpret_cast<const char *>(msg.data() + off), static_cast<int>(std::min<size_t>(msg.size() - off, 1 << 20)), 0);
        if (n <= 0) break;
        off += n;
      }
      if (off < msg.size()) break;
    }
    connected_ = false;
    uintptr_t s = sock_.exchange(INVALID_SOCKET);
    if (s != INVALID_SOCKET) {
      shutdown(sock, SD_BOTH);
      closesocket(sock);
    }
    rx.join();
    {
      std::lock_guard lk(sendMutex_);
      hasPending_ = false;
      pending_.clear();
    }
    log_("disconnected from the bridge");
  }
  WSACleanup();
}
