#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
struct Image { uint32_t width = 0, height = 0, bits = 8; std::vector<uint8_t> bytes; };
inline size_t pixel_bytes(uint32_t w, uint32_t h) {
    if (!w || !h || uint64_t(w)*h > PAINTFE_HEIC_MAX_PIXELS) throw std::runtime_error("HEIC image exceeds the configured PaintFE pixel limit");
    return size_t(w)*h*4;
}
Image decode(const uint8_t*, size_t, bool, bool);
Image encode(const uint8_t*, uint32_t, uint32_t, uint8_t, uint8_t);
uint8_t capabilities();
void limits();
