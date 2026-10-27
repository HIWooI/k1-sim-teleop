// Minimal stand-in for cppzmq's <zmq.hpp> so the official zmq_packed_message_subscriber.hpp can be compiled and
// run without libzmq (e.g. on macOS). A SUB socket "receives" the records of a capture file
// ($ZMQ_SHIM_REPLAY: repeated [u32 little-endian length][message bytes], written by linux/zmq_v3_compat.py --save),
// honoring the subscribe() prefix filter like a real SUB socket. Only the API surface used by that header exists.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#define ZMQ_REPLAY_SHIM 1

namespace zmq {
class error_t : public std::exception {
 public:
  const char* what() const noexcept override { return "zmq replay shim error"; }
};
class context_t {
 public:
  explicit context_t(int = 1) {}
};
enum class socket_type { sub, pub };
enum class recv_flags { none, dontwait };
using recv_result_t = std::optional<size_t>;
namespace sockopt {
struct rcvtimeo_t {}; struct linger_t {}; struct rcvhwm_t {}; struct conflate_t {}; struct subscribe_t {};
inline constexpr rcvtimeo_t rcvtimeo{}; inline constexpr linger_t linger{}; inline constexpr rcvhwm_t rcvhwm{};
inline constexpr conflate_t conflate{}; inline constexpr subscribe_t subscribe{};
}  // namespace sockopt
class message_t {
 public:
  std::vector<uint8_t> buf;
  const void* data() const { return buf.data(); }
  void* data() { return buf.data(); }
  size_t size() const { return buf.size(); }
};
namespace shim {
inline size_t& records_read() { static size_t n = 0; return n; }
}
class socket_t {
 public:
  socket_t(context_t&, socket_type) {}
  ~socket_t() { if (f_) std::fclose(f_); }
  template <class O, class V> void set(O, const V&) {}
  void set(sockopt::subscribe_t, const std::string& prefix) { prefix_ = prefix; }
  void connect(const std::string&) {
    const char* p = std::getenv("ZMQ_SHIM_REPLAY");
    if (!p || !(f_ = std::fopen(p, "rb"))) throw error_t();
  }
  recv_result_t recv(message_t& m, recv_flags = recv_flags::none) {
    for (;;) {
      uint32_t n = 0;
      if (!f_ || std::fread(&n, 4, 1, f_) != 1) return std::nullopt;  // EOF behaves like a receive timeout
      m.buf.resize(n);
      if (n && std::fread(m.buf.data(), 1, n, f_) != n) return std::nullopt;
      shim::records_read()++;
      if (prefix_.empty() || (n >= prefix_.size() && std::memcmp(m.buf.data(), prefix_.data(), prefix_.size()) == 0)) return n;
    }
  }
  void close() {}
 private:
  std::FILE* f_ = nullptr;
  std::string prefix_;
};
}  // namespace zmq
