// Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
// licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
// Decode SONIC ZMQ Protocol v3 messages with the *official* deploy subscriber header.
//
//   ZMQPackedMessageSubscriber (gear_sonic_deploy/.../input_interface/zmq_packed_message_subscriber.hpp)
//   receives each message, strips the topic, parses the 1280-byte JSON header and slices the fields, exactly
//   as the deploy binary does. The callback then applies the v3 checks of
//   ZMQEndpointInterface::DecodeIntoMotionSequence (field names/aliases, required fields, shapes,
//   frame counts, f32/f64 decode, frame_index i32/i64, timestamp_monotonic f64).
//
// Linux (real libzmq + cppzmq), against a live sender:
//   zmq_v3_check --host <sender-ip> [--port 5556] [--topic pose] [--count 300]
// macOS / no libzmq: build with -I linux/compat/zmq_replay_shim and replay a capture file
//   ZMQ_SHIM_REPLAY=capture.bin zmq_v3_check --count 1000000
// (capture files come from `linux/zmq_v3_compat.py e2e|listen|selftest --save FILE`).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "zmq_packed_message_subscriber.hpp"

using Sub = ZMQPackedMessageSubscriber;

struct Stats {
  int received = 0, ok = 0, rejected = 0, resets = 0, gaps = 0;
  int version = -1;
  int64_t last_frame = -1;
  std::map<std::string, int> reasons;
};

static bool reject(Stats& s, const std::string& why) {
  s.rejected++;
  s.reasons[why]++;
  return false;
}

template <typename T>
static double rd(const Sub::BufferView& b, size_t i) {
  T v;
  std::memcpy(&v, static_cast<const uint8_t*>(b.data) + i * sizeof(T), sizeof(T));
  return static_cast<double>(v);
}
static double rdf(const Sub::FieldInfo& f, const Sub::BufferView& b, size_t i) {
  return f.dtype == "f64" ? rd<double>(b, i) : rd<float>(b, i);
}

static bool check_v3(Stats& s, const Sub::DecodedHeader& h, const std::vector<Sub::BufferView>& bufs, bool print) {
  // Field lookup — same names and aliases as ZMQEndpointInterface::DecodeIntoMotionSequence.
  int jp = -1, jv = -1, bq = -1, fi = -1, sj = -1, sp = -1, ts = -1;
  for (size_t i = 0; i < h.fields.size(); ++i) {
    const auto& n = h.fields[i].name;
    if (n == "joint_pos") jp = int(i);
    else if (n == "joint_vel") jv = int(i);
    else if (n == "body_quat_w" || n == "body_quat") bq = int(i);
    else if (n == "frame_index" || n == "last_smpl_global_frames") fi = int(i);
    else if (n == "smpl_joints") sj = int(i);
    else if (n == "smpl_pose") sp = int(i);
    else if (n == "timestamp_monotonic") ts = int(i);
  }
  if (h.version != 3) return reject(s, "protocol version " + std::to_string(h.version) + " (expected 3)");
  if (bq < 0) return reject(s, "Missing required field 'body_quat'");
  if (fi < 0) return reject(s, "Missing required field 'frame_index'");
  if (sj < 0 || sp < 0) return reject(s, "Version 3 missing smpl_joints / smpl_pose");
  if (jp < 0 || jv < 0) return reject(s, "Version 3 missing joint_pos / joint_vel");
  const auto &fsj = h.fields[sj], &fjp = h.fields[jp], &fjv = h.fields[jv], &fbq = h.fields[bq], &ffi = h.fields[fi];
  if (fsj.shape.size() < 2 || fsj.shape[0] == 0) return reject(s, "Invalid smpl_joints shape");
  if (fjp.shape.size() != 2 || fjv.shape.size() != 2) return reject(s, "invalid joint_pos/joint_vel shape");
  const size_t n = fsj.shape[0];
  if (fjp.shape[0] != n || fjv.shape[0] != n) return reject(s, "Version 3 frame count mismatch");
  for (const auto* f : {&fsj, &fjp, &fjv, &fbq, &h.fields[sp]})
    if (f->dtype != "f32" && f->dtype != "f64") return reject(s, f->name + " dtype " + f->dtype + " not decoded by deploy");
  if (ffi.dtype != "i32" && ffi.dtype != "i64") return reject(s, "frame_index dtype " + ffi.dtype + " -> no frame indices");
  if (s.version >= 0 && s.version != h.version) return reject(s, "protocol version changed mid-session");
  s.version = h.version;

  const int64_t frame = ffi.dtype == "i64" ? static_cast<int64_t>(rd<int64_t>(bufs[fi], 0)) : static_cast<int64_t>(rd<int32_t>(bufs[fi], 0));
  if (s.last_frame >= 0) {
    if (frame <= s.last_frame) s.resets++;  // merger: incoming_frame_start <= window start -> catch-up reset
    else if (frame > s.last_frame + 1) s.gaps++;
  }
  s.last_frame = frame;

  const size_t nsj = fsj.shape.size() == 3 ? fsj.shape[1] : 1;
  double qn = 0;
  for (int q = 0; q < 4; ++q) qn += std::pow(rdf(fbq, bufs[bq], q), 2);
  if (std::fabs(std::sqrt(qn) - 1.0) > 1e-3) return reject(s, "body_quat not unit norm");
  if (print) {
    std::cout << std::fixed << std::setprecision(4) << "  v" << h.version << " frame_index=" << frame
              << " smpl_joints[" << n << "," << nsj << ",3] joint_pos[" << n << "," << fjp.shape[1] << "]"
              << " body_quat=(" << rdf(fbq, bufs[bq], 0) << "," << rdf(fbq, bufs[bq], 1) << "," << rdf(fbq, bufs[bq], 2)
              << "," << rdf(fbq, bufs[bq], 3) << ") pelvis=(" << rdf(fsj, bufs[sj], 0) << "," << rdf(fsj, bufs[sj], 1)
              << "," << rdf(fsj, bufs[sj], 2) << ")";
    if (ts >= 0 && h.fields[ts].dtype == "f64") std::cout << " t_mono=" << std::setprecision(3) << rd<double>(bufs[ts], 0);
    std::cout << std::endl;
  }
  s.ok++;
  return true;
}

int main(int argc, char** argv) {
  std::string host = "localhost", topic = "pose";
  int port = 5556, count = 300, timeout_ms = 1000, idle_limit = 10;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](void) -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "--topic") topic = next();
    else if (a == "--count") count = std::stoi(next());
    else if (a == "--idle") idle_limit = std::stoi(next());
    else { std::cerr << "usage: zmq_v3_check [--host H] [--port P] [--topic T] [--count N] [--idle SECONDS]\n"; return 2; }
  }
  Stats s;
  Sub sub(host, port, topic, timeout_ms, /*verbose=*/false, /*conflate=*/false, /*rcv_hwm=*/3);  // deploy's settings
  sub.SetOnDecodedMessage([&](const std::string&, const Sub::DecodedHeader& h, const std::vector<Sub::BufferView>& b) {
    s.received++;
    check_v3(s, h, b, s.received <= 3);
  });
  if (!sub.Connect()) return 2;
  std::cout << "[zmq_v3_check] official ZMQPackedMessageSubscriber, tcp://" << host << ":" << port << " topic '" << topic << "'"
            << std::endl;
  int idle = 0;
  while (s.received < count && idle < idle_limit) {
    if (sub.PollOnce()) idle = 0;
    else idle++;  // PollOnce returns false on timeout (1 s) or on a frame the official parser dropped
  }
  std::cout << "[zmq_v3_check] received=" << s.received << " v3_ok=" << s.ok << " rejected=" << s.rejected
            << " frame_resets=" << s.resets << " frame_gaps=" << s.gaps << std::endl;
  for (const auto& [why, k] : s.reasons) std::cout << "  rejected x" << k << ": " << why << std::endl;
#ifdef ZMQ_REPLAY_SHIM
  // every replayed record must have reached the callback, i.e. none was dropped by the official parser
  std::cout << "[zmq_v3_check] replayed records=" << zmq::shim::records_read() << std::endl;
  if (zmq::shim::records_read() != static_cast<size_t>(s.received)) s.rejected++;
#endif
  const bool pass = s.received > 0 && s.rejected == 0 && s.resets == 0;
  std::cout << "RESULT " << (pass ? "COMPATIBLE" : "INCOMPATIBLE") << std::endl;
  return pass ? 0 : 1;
}
