#pragma once
#include "bridge.hpp"
#include <memory>
#ifdef _WIN32
#include <windows.h>
#include <filesystem>
#else
#include <dlfcn.h>
#endif
// Shared runtime adapter. Codec binaries are supplied by the system or user.
namespace runtime_heif {
struct Error { int code, subcode; const char* message; };
struct Writer { int version; Error (*write)(void*,const void*,size_t,void*); };
struct Library {
#ifdef _WIN32
    HMODULE lib=nullptr;
    Library() {
        const wchar_t* configured=_wgetenv(L"PAINTFE_HEIF_LIBRARY");
        if(!configured || !*configured) throw std::runtime_error("Choose a libheif DLL in Optional Features first");
        std::filesystem::path path(configured);
        if(!path.is_absolute()) throw std::runtime_error("The libheif path must be absolute");
        lib=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if(!lib) throw std::runtime_error("Could not load libheif. Check that the DLL and all its dependencies match PaintFE's architecture.");
        auto version=get<unsigned(*)()>("heif_get_version_number")();
        if((version>>24)!=1 || version<0x010d0000) { FreeLibrary(lib); lib=nullptr; throw std::runtime_error("Use a compatible libheif 1.x library, version 1.13 or newer"); }
        auto init=get<Error(*)(void*)>("heif_init")(nullptr);
        if(init.code) { FreeLibrary(lib); lib=nullptr; throw std::runtime_error("libheif plugin initialization failed"); }
    }
    ~Library(){ if(lib) { get<void(*)()>("heif_deinit")(); FreeLibrary(lib); } }
    template<class T> T get(const char* name) {
        auto p=GetProcAddress(lib,name); if(!p) throw std::runtime_error(std::string("libheif is missing ")+name);
        return reinterpret_cast<T>(p);
    }
#else
    void* lib=dlopen("libheif.so.1",RTLD_NOW|RTLD_LOCAL);
    Library(){ if(!lib) throw std::runtime_error("HEIC requires a system libheif installation with an HEVC codec. Install it using your distribution's package manager; PaintFE does not download HEVC codecs."); }
    ~Library(){dlclose(lib);}
    template<class T> T get(const char* name) {
        auto p=dlsym(lib,name); if(!p) throw std::runtime_error(std::string("System libheif is missing ")+name);
        return reinterpret_cast<T>(p);
    }
#endif
};
static void check(Error error) { if(error.code) throw std::runtime_error(std::string("System HEIC codec: ")+(error.message?error.message:"failed")); }
struct Owned {
    void* value=nullptr; void (*release)(void*);
    explicit Owned(void (*r)(void*)):release(r){} ~Owned(){if(value) release(value);}
};
uint8_t capabilities() {
    Library l;
    auto decoder=l.get<int(*)(int)>("heif_have_decoder_for_format");
    auto encoder=l.get<int(*)(int)>("heif_have_encoder_for_format");
    uint8_t flags=(decoder(1)?1:0)|(encoder(1)?2:0);
    const void* descriptor=nullptr;
    try {
    if(encoder(1) && l.get<int(*)(int,const char*,const void**,int)>("heif_get_encoder_descriptors")(1,nullptr,&descriptor,1)>0) {
        if(l.get<int(*)(const void*)>("heif_encoder_descriptor_supports_lossless_compression")(descriptor)) flags|=4;
        Owned selected(l.get<void(*)(void*)>("heif_encoder_release"));
        Owned ctx(l.get<void(*)(void*)>("heif_context_free"));ctx.value=l.get<void*(*)()>("heif_context_alloc")();
        if(!ctx.value) throw std::bad_alloc();
        check(l.get<Error(*)(void*,const void*,void**)>("heif_context_get_encoder")(ctx.value,descriptor,&selected.value));
        const char* const* values=nullptr;
        auto error=l.get<Error(*)(void*,const char*,const char* const**)>("heif_encoder_parameter_string_valid_values")(selected.value,"chroma",&values);
        if(!error.code && values) for(size_t i=0;i<32 && values[i];i++) {
            if(!std::strcmp(values[i],"444")) flags|=8;
            if(!std::strcmp(values[i],"422")) flags|=16;
            if(!std::strcmp(values[i],"420")) flags|=32;
        }
    }
    } catch(const std::runtime_error&) { /* Optional capabilities must not disable basic decoding. */ }
    return flags;
}
Image decode(const uint8_t* data,size_t size,bool requires_alpha,bool high_depth) {
    Library l;
    Owned ctx(l.get<void(*)(void*)>("heif_context_free")); ctx.value=l.get<void*(*)()>("heif_context_alloc")();
    if(!ctx.value) throw std::bad_alloc();
    check(l.get<Error(*)(void*,const void*,size_t,void*)>("heif_context_read_from_memory_without_copy")(ctx.value,data,size,nullptr));
    Owned handle(l.get<void(*)(void*)>("heif_image_handle_release"));
    check(l.get<Error(*)(void*,void**)>("heif_context_get_primary_image_handle")(ctx.value,&handle.value));
    if(requires_alpha && !l.get<int(*)(const void*)>("heif_image_handle_has_alpha_channel")(handle.value))
        throw std::runtime_error("System HEIC decoder cannot preserve this image's transparency");
    auto width=l.get<int(*)(const void*)>("heif_image_handle_get_width");
    auto height=l.get<int(*)(const void*)>("heif_image_handle_get_height");
    Image out; out.width=width(handle.value); out.height=height(handle.value);
    out.bits=high_depth?16:8;
    out.bytes.resize(pixel_bytes(out.width,out.height)*(out.bits/8));
    Owned image(l.get<void(*)(void*)>("heif_image_release"));
    check(l.get<Error(*)(const void*,void**,int,int,const void*)>("heif_decode_image")(handle.value,&image.value,1,high_depth?15:11,nullptr));
    int stride=0; const auto* pixels=l.get<const uint8_t*(*)(const void*,int,int*)>("heif_image_get_plane_readonly")(image.value,10,&stride);
    auto row=out.width*4*(out.bits/8);
    if(!pixels || stride<int(row)) throw std::runtime_error("Invalid system HEIC pixel plane");
    for(uint32_t y=0;y<out.height;y++) std::memcpy(out.bytes.data()+size_t(y)*row,pixels+size_t(y)*stride,row);
    if(high_depth) {
        int depth=l.get<int(*)(const void*,int)>("heif_image_get_bits_per_pixel_range")(image.value,10);
        if(depth<9 || depth>16) throw std::runtime_error("System HEIC decoder did not preserve high-bit-depth precision");
        uint32_t maximum=(1u<<depth)-1;
        for(size_t i=0;i<out.bytes.size();i+=2) {
            uint32_t value=out.bytes[i]|(uint32_t(out.bytes[i+1])<<8);
            value=(std::min(value,maximum)*65535+maximum/2)/maximum;
            out.bytes[i]=value&255; out.bytes[i+1]=value>>8;
        }
    }
    return out;
}
static Error write_bytes(void*,const void* data,size_t size,void* user) noexcept {
    auto& bytes=*static_cast<std::vector<uint8_t>*>(user);
    try {
        if(size>512*1024*1024 || bytes.size()>512*1024*1024-size) return {9,0,"HEIC output exceeds limit"};
        auto p=static_cast<const uint8_t*>(data); bytes.insert(bytes.end(),p,p+size); return {0,0,"Success"};
    } catch(...) {return {9,0,"HEIC output allocation failed"};}
}
Image encode(const uint8_t* data,uint32_t w,uint32_t h,uint8_t quality,uint8_t options) {
    Library l;
    Owned ctx(l.get<void(*)(void*)>("heif_context_free")); ctx.value=l.get<void*(*)()>("heif_context_alloc")();
    if(!ctx.value) throw std::bad_alloc();
    Owned encoder(l.get<void(*)(void*)>("heif_encoder_release"));
    check(l.get<Error(*)(void*,int,void**)>("heif_context_get_encoder_for_format")(ctx.value,1,&encoder.value));
    check(l.get<Error(*)(void*,int)>("heif_encoder_set_lossy_quality")(encoder.value,quality));
    if(options&1) check(l.get<Error(*)(void*,int)>("heif_encoder_set_lossless")(encoder.value,1));
    int chroma=(options>>1)&3;
    if(options&1) chroma=1;
    if(chroma) check(l.get<Error(*)(void*,const char*,const char*)>("heif_encoder_set_parameter_string")(encoder.value,"chroma",chroma==1?"444":chroma==2?"422":"420"));
    // Encoders expose different parameter sets; use the standard thread control
    // when supported without requiring a particular HEVC implementation.
    l.get<Error(*)(void*,const char*,int)>("heif_encoder_set_parameter_integer")(encoder.value,"threads",2);
    Owned image(l.get<void(*)(void*)>("heif_image_release"));
    check(l.get<Error(*)(int,int,int,int,void**)>("heif_image_create")(w,h,1,11,&image.value));
    check(l.get<Error(*)(void*,int,int,int,int)>("heif_image_add_plane")(image.value,10,w,h,8));
    int stride=0; auto* pixels=l.get<uint8_t*(*)(void*,int,int*)>("heif_image_get_plane")(image.value,10,&stride);
    if(!pixels || stride<int(w*4)) throw std::runtime_error("Invalid system HEIC output plane");
    for(uint32_t y=0;y<h;y++) std::memcpy(pixels+size_t(y)*stride,data+size_t(y)*w*4,w*4);
    // Stable version-1 NCLX prefix. Identity, full-range RGB avoids conversion loss.
    struct NclxPrefix {uint8_t version; int primaries,transfer,matrix; uint8_t full_range;};
    Owned nclx(l.get<void(*)(void*)>("heif_nclx_color_profile_free"));
    if(options&1) {
        nclx.value=l.get<void*(*)()>("heif_nclx_color_profile_alloc")();
        if(!nclx.value) throw std::bad_alloc();
        auto* profile=static_cast<NclxPrefix*>(nclx.value);
        profile->primaries=1;profile->transfer=13;profile->matrix=0;profile->full_range=1;
        check(l.get<Error(*)(void*,const void*)>("heif_image_set_nclx_color_profile")(image.value,nclx.value));
    }
    check(l.get<Error(*)(void*,const void*,void*,const void*,void**)>("heif_context_encode_image")(ctx.value,image.value,encoder.value,nullptr,nullptr));
    Image out; Writer writer{1,write_bytes};
    check(l.get<Error(*)(void*,const Writer*,void*)>("heif_context_write")(ctx.value,&writer,&out.bytes));
    return out;
}

}
