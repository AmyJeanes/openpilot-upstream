// TCP link to the bridge. The plugin connects out: under WSL, the bridge is reachable at the distro's IP, while Windows
// usually firewalls connections the other way.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Message = std::map<std::string, std::string>;  // a flat JSON object; values keep their JSON text minus string quotes

bool ParseFlatJson(const std::string &text, Message &out);
double MsgNum(const Message &m, const char *key, double fallback = 0.0);
bool MsgBool(const Message &m, const char *key, bool fallback = false);
std::string MsgStr(const Message &m, const char *key, const std::string &fallback = "");

class Link {
 public:
  // address() returns "host:port" or "" when unknown; it is asked again before each connection attempt
  void Start(std::function<std::string()> address, std::function<void(const std::string &)> log);
  void Stop();
  bool Connected() const { return connected_; }

  // Queues a frame message: [u32 total][u32 header length][header JSON][payload]. Only the newest frame is kept while
  // the previous one is still being sent.
  void SendFrame(const std::string &header, std::vector<uint8_t> &&payload);
  std::vector<Message> TakeMessages();

 private:
  void Run();
  void Receive(uintptr_t sock);

  std::function<std::string()> address_;
  std::function<void(const std::string &)> log_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> connected_{false};
  std::atomic<uintptr_t> sock_{~uintptr_t(0)};

  std::mutex sendMutex_;
  std::condition_variable sendCv_;
  std::vector<uint8_t> pending_;
  bool hasPending_ = false;

  std::mutex recvMutex_;
  std::deque<Message> received_;
};
