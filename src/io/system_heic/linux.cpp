#include "libheif_codec.hpp"
#include <sys/resource.h>
uint8_t capabilities() { return runtime_heif::capabilities(); }
Image decode(const uint8_t* data,size_t size,bool alpha,bool high_depth) { return runtime_heif::decode(data,size,alpha,high_depth); }
Image encode(const uint8_t* data,uint32_t w,uint32_t h,uint8_t quality,uint8_t options) { return runtime_heif::encode(data,w,h,quality,options); }
void limits() {
    struct rlimit current{};
    if(getrlimit(RLIMIT_AS,&current)) throw std::runtime_error("Could not read system codec memory limit");
    current.rlim_cur=current.rlim_max=std::min(current.rlim_max,rlim_t(3)*1024*1024*1024);
    if(setrlimit(RLIMIT_AS,&current)) throw std::runtime_error("Could not apply system codec memory limit");
}
