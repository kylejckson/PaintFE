// SPDX-License-Identifier: GPL-3.0-or-later
// Offline HEIC export helper. The signed pack includes corresponding source.
#include <libheif/heif.h>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <thread>
#include <chrono>
#else
#include <sys/resource.h>
#endif

static void check(heif_error error) {
  if (error.code != heif_error_Ok) throw std::runtime_error(error.message);
}
static uint32_t field(const unsigned char* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
static heif_error write_bytes(heif_context*, const void* data, size_t size, void* user) {
  auto& output = *static_cast<std::vector<unsigned char>*>(user);
  if (size > 256u * 1024 * 1024 - output.size())
    return {heif_error_Encoding_error, heif_suberror_Unspecified, "HEIC output exceeds limit"};
  try { auto p = static_cast<const unsigned char*>(data); output.insert(output.end(), p, p + size); }
  catch (...) { return {heif_error_Memory_allocation_error, heif_suberror_Unspecified, "HEIC allocation failed"}; }
  return {heif_error_Ok, heif_suberror_Unspecified, "OK"};
}
static void limit_memory() {
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY);
  static HANDLE job = CreateJobObjectW(nullptr, nullptr);
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
  limits.ProcessMemoryLimit = size_t(2) * 1024 * 1024 * 1024;
  if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) || !AssignProcessToJobObject(job, GetCurrentProcess()))
    throw std::runtime_error("Cannot apply HEIC memory limit");
#elif defined(__APPLE__)
  std::thread([] { for (;;) { mach_task_basic_info_data_t info{}; mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS || info.resident_size > uint64_t(2) * 1024 * 1024 * 1024) std::_Exit(3);
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); } }).detach();
#else
  rlimit limits{}; if (getrlimit(RLIMIT_AS, &limits)) throw std::runtime_error("Cannot inspect HEIC memory limit");
  auto cap = rlim_t(3) * 1024 * 1024 * 1024;
  limits.rlim_cur = std::min(limits.rlim_cur, cap); limits.rlim_max = std::min(limits.rlim_max, cap);
  if (setrlimit(RLIMIT_AS, &limits)) throw std::runtime_error("Cannot apply HEIC memory limit");
#endif
}
int main() {
  try {
    limit_memory();
    std::array<unsigned char, 40> header{};
    if (!std::cin.read(reinterpret_cast<char*>(header.data()), header.size()) || std::memcmp(header.data(), "PFEHIC01", 8) || header[8] != 1)
      throw std::runtime_error("Invalid HEIC request");
    const auto w = field(&header[12]), h = field(&header[16]);
    const uint64_t pixels = uint64_t(w) * h, length = uint64_t(field(&header[32])) | uint64_t(field(&header[36])) << 32;
    if (!w || !h || pixels > 64000000 || length != pixels * 4 || field(&header[20]) != 8 || header[24] < 1 || header[24] > 100 || header[25] != 8 || header[9] || header[10] || header[11] || header[26] || header[27] || field(&header[28]))
      throw std::runtime_error("Invalid HEIC dimensions, length or settings");
    std::vector<unsigned char> rgba(static_cast<size_t>(length));
    if (!std::cin.read(reinterpret_cast<char*>(rgba.data()), rgba.size()) || std::cin.peek() != EOF)
      throw std::runtime_error("Incomplete or trailing HEIC pixels");
    check(heif_init(nullptr));
    auto context = std::unique_ptr<heif_context, decltype(&heif_context_free)>(heif_context_alloc(), heif_context_free);
    if (!context) throw std::bad_alloc();
    heif_encoder* encoder_raw = nullptr;
    check(heif_context_get_encoder_for_format(context.get(), heif_compression_HEVC, &encoder_raw));
    auto encoder = std::unique_ptr<heif_encoder, decltype(&heif_encoder_release)>(encoder_raw, heif_encoder_release);
    check(heif_encoder_set_lossy_quality(encoder.get(), header[24]));
    heif_image* image_raw = nullptr;
    check(heif_image_create(w, h, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, &image_raw));
    auto image = std::unique_ptr<heif_image, decltype(&heif_image_release)>(image_raw, heif_image_release);
    check(heif_image_add_plane(image.get(), heif_channel_interleaved, w, h, 8));
    int stride = 0; auto plane = heif_image_get_plane(image.get(), heif_channel_interleaved, &stride);
    if (!plane || stride < int(w * 4)) throw std::runtime_error("Invalid HEIC image plane");
    for (uint32_t y = 0; y < h; ++y) std::memcpy(plane + size_t(y) * stride, rgba.data() + size_t(y) * w * 4, size_t(w) * 4);
    auto options = std::unique_ptr<heif_encoding_options, decltype(&heif_encoding_options_free)>(heif_encoding_options_alloc(), heif_encoding_options_free);
    if (!options) throw std::bad_alloc();
    options->save_alpha_channel = 1;
    check(heif_context_encode_image(context.get(), image.get(), encoder.get(), options.get(), nullptr));
    std::vector<unsigned char> output; heif_writer writer{1, write_bytes};
    check(heif_context_write(context.get(), &writer, &output));
    std::cout.write(reinterpret_cast<const char*>(output.data()), output.size()); std::cout.flush();
    if (!std::cout) throw std::runtime_error("HEIC output failed");
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
  catch (...) { std::cerr << "HEIC codec failed\n"; return 2; }
}
