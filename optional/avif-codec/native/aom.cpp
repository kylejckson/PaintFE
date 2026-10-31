#include <aom/aom_encoder.h>
#include <aom/aomcx.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Codec {
    aom_codec_ctx_t context{};
    bool initialized=false;
    ~Codec(){if(initialized) aom_codec_destroy(&context);}
};
void check(aom_codec_err_t status) {
    if(status!=AOM_CODEC_OK) throw std::runtime_error(aom_codec_err_to_string(status));
}
std::vector<uint8_t> encode(const uint8_t* rgba,unsigned width,unsigned height,
                           unsigned quality,unsigned speed,bool lossless,bool alpha) {
    aom_codec_enc_cfg_t config{};
    check(aom_codec_enc_config_default(aom_codec_av1_cx(),&config,AOM_USAGE_ALL_INTRA));
    config.g_w=width;config.g_h=height;config.g_threads=2;config.g_lag_in_frames=0;
    config.g_profile=alpha?0:1;config.g_bit_depth=AOM_BITS_8;config.g_input_bit_depth=8;
    config.monochrome=alpha?1:0;config.rc_end_usage=AOM_Q;
    config.rc_min_quantizer=config.rc_max_quantizer=lossless?0:((100-quality)*63+49)/99;
    Codec codec;check(aom_codec_enc_init(&codec.context,aom_codec_av1_cx(),&config,0));codec.initialized=true;
    check(aom_codec_control(&codec.context,AOME_SET_CPUUSED,static_cast<int>(speed-1)));
    check(aom_codec_control(&codec.context,AOME_SET_TUNING,AOM_TUNE_PSNR));
    check(aom_codec_control(&codec.context,AV1E_SET_ROW_MT,1));
    check(aom_codec_control(&codec.context,AV1E_SET_ENABLE_CHROMA_DELTAQ,0));
    check(aom_codec_control(&codec.context,AV1E_SET_LOSSLESS,lossless?1:0));
    check(aom_codec_control(&codec.context,AV1E_SET_COLOR_RANGE,AOM_CR_FULL_RANGE));
    if(!alpha) {
        check(aom_codec_control(&codec.context,AV1E_SET_COLOR_PRIMARIES,AOM_CICP_CP_BT_709));
        check(aom_codec_control(&codec.context,AV1E_SET_TRANSFER_CHARACTERISTICS,AOM_CICP_TC_SRGB));
        check(aom_codec_control(&codec.context,AV1E_SET_MATRIX_COEFFICIENTS,AOM_CICP_MC_IDENTITY));
    }
    std::unique_ptr<aom_image_t,decltype(&aom_img_free)> image(
        aom_img_alloc(nullptr,alpha?AOM_IMG_FMT_I420:AOM_IMG_FMT_I444,width,height,1),aom_img_free);
    if(!image) throw std::bad_alloc();
    image->range=AOM_CR_FULL_RANGE;
    for(unsigned y=0;y<height;y++) for(unsigned x=0;x<width;x++) {
        auto* p=rgba+(size_t(y)*width+x)*4;
        image->planes[0][size_t(y)*image->stride[0]+x]=p[alpha?3:1];
        if(!alpha) {
            image->planes[1][size_t(y)*image->stride[1]+x]=p[2];
            image->planes[2][size_t(y)*image->stride[2]+x]=p[0];
        }
    }
    if(alpha) for(int c=1;c<3;c++) for(unsigned y=0;y<(height+1)/2;y++)
        std::memset(image->planes[c]+size_t(y)*image->stride[c],128,(width+1)/2);
    check(aom_codec_encode(&codec.context,image.get(),0,1,AOM_EFLAG_FORCE_KF));
    std::vector<uint8_t> output;
    auto collect=[&] {
        aom_codec_iter_t iterator=nullptr;
        while(auto* packet=aom_codec_get_cx_data(&codec.context,&iterator)) {
            if(packet->kind!=AOM_CODEC_CX_FRAME_PKT) continue;
            if(packet->data.frame.sz>512*1024*1024-output.size()) throw std::runtime_error("AVIF output exceeds limit");
            const auto* bytes=static_cast<const uint8_t*>(packet->data.frame.buf);
            output.insert(output.end(),bytes,bytes+packet->data.frame.sz);
        }
    };
    collect();check(aom_codec_encode(&codec.context,nullptr,1,1,0));collect();
    if(output.empty()) throw std::runtime_error("AVIF encoder produced no image");
    return output;
}
}
extern "C" int pf_aom_encode(const uint8_t* rgba,uint32_t width,uint32_t height,uint8_t quality,
    uint8_t speed,uint8_t flags,uint8_t** color,size_t* color_size,uint8_t** alpha,size_t* alpha_size,
    char* error,size_t error_size) noexcept {
    *color=nullptr;*alpha=nullptr;*color_size=0;*alpha_size=0;
    try {
        if(!rgba || !width || !height || uint64_t(width)*height>64000000 || quality<1 || quality>100 || speed<1 || speed>10 || flags>3)
            throw std::runtime_error("Invalid AVIF encoder input");
        auto rgb=encode(rgba,width,height,quality,speed,(flags&1)!=0,false);
        bool transparent=false;
        for(size_t i=3;i<size_t(width)*height*4;i+=4) if(rgba[i]!=255){transparent=true;break;}
        std::vector<uint8_t> mask;
        if(transparent) mask=encode(rgba,width,height,quality,speed,flags!=0,true);
        *color=static_cast<uint8_t*>(std::malloc(rgb.size()));
        if(!*color) throw std::bad_alloc();
        std::memcpy(*color,rgb.data(),rgb.size());*color_size=rgb.size();
        if(!mask.empty()) {
            *alpha=static_cast<uint8_t*>(std::malloc(mask.size()));
            if(!*alpha) throw std::bad_alloc();
            std::memcpy(*alpha,mask.data(),mask.size());*alpha_size=mask.size();
        }
        return 0;
    } catch(const std::exception& e) {
        std::free(*color);std::free(*alpha);*color=nullptr;*alpha=nullptr;
        if(error_size){std::strncpy(error,e.what(),error_size-1);error[error_size-1]=0;}
        return 1;
    } catch(...) {return 1;}
}
extern "C" void pf_aom_free(uint8_t* pointer) noexcept {std::free(pointer);}
