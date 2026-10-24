#include "bridge.hpp"
extern "C" int pf_system_heic(uint8_t op, const uint8_t* data, size_t length,
    uint32_t width, uint32_t height, uint8_t quality, uint8_t** output, size_t* size,
    uint32_t* out_width, uint32_t* out_height, uint32_t* out_bits, char* error, size_t error_size) noexcept {
    *output = nullptr; *size = 0;
    const bool requires_alpha=(op&0x80)!=0;
    const bool high_depth=(op&0x40)!=0;
    op&=0x3f;
    try {
        Image result;
        if (op == 0) result.bytes = {capabilities()};
        else if (op == 1) {
            if (length != pixel_bytes(width,height)) throw std::runtime_error("Invalid HEIC input length");
            result = encode(data,width,height,quality);
            bool alpha=false;
            for(size_t i=3;i<length;i+=4) if(data[i]!=255) {alpha=true;break;}
            if(alpha) {
                auto check_image=decode(result.bytes.data(),result.bytes.size(),true,false);
                bool valid=check_image.width==width && check_image.height==height && check_image.bytes.size()==length;
                if(valid) for(size_t i=3;i<length;i+=4) if(std::abs(int(check_image.bytes[i])-int(data[i]))>1) {valid=false;break;}
                if(!valid) throw std::runtime_error("The installed HEIC encoder cannot preserve transparency. Save as PNG or PFE instead.");
            }
        } else if (op == 2) {
            if (length < 16 || length > 256*1024*1024) throw std::runtime_error("Invalid HEIC input size");
            result = decode(data,length,requires_alpha,high_depth);
            if ((result.bits!=8 && result.bits!=16) || result.bytes.size() != pixel_bytes(result.width,result.height)*(result.bits/8)) throw std::runtime_error("Invalid HEIC pixels");
        } else throw std::runtime_error("Unknown HEIC operation");
        if (result.bytes.size() > 512*1024*1024) throw std::runtime_error("HEIC output exceeds limit");
        *output = static_cast<uint8_t*>(std::malloc(result.bytes.size()));
        if (!*output) throw std::bad_alloc();
        std::memcpy(*output,result.bytes.data(),result.bytes.size());
        *size = result.bytes.size(); *out_width = result.width; *out_height = result.height; *out_bits=result.bits;
        return 0;
    } catch (const std::exception& e) {
        if (error_size) { std::strncpy(error,e.what(),error_size-1); error[error_size-1] = 0; }
        return 1;
    } catch (...) {
        if (error_size) { std::strncpy(error,"System HEIC codec failed",error_size-1); error[error_size-1] = 0; }
        return 1;
    }
}
extern "C" void pf_system_heic_free(uint8_t* bytes) noexcept { std::free(bytes); }
extern "C" int pf_system_heic_limits(char* error,size_t capacity) noexcept {
    try { limits(); return 0; }
    catch(const std::exception& e) { if(capacity) {std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;} return 1; }
}
