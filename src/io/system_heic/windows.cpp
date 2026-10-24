#define NOMINMAX
#include "bridge.hpp"
#include <windows.h>
#include <wincodec.h>
#include <shlwapi.h>
#include <wrl/client.h>
#include <mfapi.h>
#include <mfidl.h>
#include <cstdio>
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr) {
    if (FAILED(hr)) { char code[32]; std::snprintf(code,sizeof(code)," (0x%08lX)",static_cast<unsigned long>(hr));
        throw std::runtime_error(std::string("Windows HEIC codec failed")+code+". The installed codec may not support this image variant or requested save options."); }
}
struct COM {
    HRESULT hr = CoInitializeEx(nullptr,COINIT_MULTITHREADED);
    COM() { if (hr != RPC_E_CHANGED_MODE) check(hr); }
    ~COM() { if (SUCCEEDED(hr)) CoUninitialize(); }
};
static ComPtr<IWICImagingFactory> factory() {
    ComPtr<IWICImagingFactory> f;
    check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&f)));
    return f;
}
uint8_t capabilities() {
    COM init; auto f = factory(); uint8_t flags = 0;
    // Media Foundation is optional on Windows N/Server. Do not put mfplat.dll
    // in the editor's import table or make non-HEIC editing depend on it.
    HMODULE media=LoadLibraryExW(L"mfplat.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(!media) return 0;
    auto startup=reinterpret_cast<decltype(&MFStartup)>(GetProcAddress(media,"MFStartup"));
    auto shutdown=reinterpret_cast<decltype(&MFShutdown)>(GetProcAddress(media,"MFShutdown"));
    auto enumerate=reinterpret_cast<decltype(&MFTEnumEx)>(GetProcAddress(media,"MFTEnumEx"));
    if(!startup || !shutdown || !enumerate || FAILED(startup(MF_VERSION,MFSTARTUP_NOSOCKET))) {FreeLibrary(media);return 0;}
    auto has_hevc=[&](bool encoder) {
        MFT_REGISTER_TYPE_INFO type{MFMediaType_Video,MFVideoFormat_HEVC};
        IMFActivate** entries=nullptr; UINT32 count=0;
        HRESULT hr=enumerate(encoder?MFT_CATEGORY_VIDEO_ENCODER:MFT_CATEGORY_VIDEO_DECODER,
            MFT_ENUM_FLAG_ALL,encoder?nullptr:&type,encoder?&type:nullptr,&entries,&count);
        for(UINT32 i=0;i<count;i++) entries[i]->Release();
        CoTaskMemFree(entries);
        return SUCCEEDED(hr) && count>0;
    };
    ComPtr<IWICBitmapDecoder> d; ComPtr<IWICBitmapEncoder> e;
    if (SUCCEEDED(f->CreateDecoder(GUID_ContainerFormatHeif,nullptr,&d)) && has_hevc(false)) flags |= 1;
    if (SUCCEEDED(f->CreateEncoder(GUID_ContainerFormatHeif,nullptr,&e)) && has_hevc(true)) flags |= 2;
    shutdown(); FreeLibrary(media);
    return flags;
}
Image decode(const uint8_t* data, size_t size, bool requires_alpha, bool high_depth) {
    COM init; auto f = factory();
    ComPtr<IWICStream> stream; check(f->CreateStream(&stream));
    check(stream->InitializeFromMemory(const_cast<BYTE*>(data),static_cast<DWORD>(size)));
    ComPtr<IWICBitmapDecoder> decoder;
    check(f->CreateDecoderFromStream(stream.Get(),nullptr,WICDecodeMetadataCacheOnDemand,&decoder));
    ComPtr<IWICBitmapFrameDecode> frame; check(decoder->GetFrame(0,&frame));
    WICPixelFormatGUID source_format; check(frame->GetPixelFormat(&source_format));
    bool combined_alpha=source_format==GUID_WICPixelFormat64bppRGBA || source_format==GUID_WICPixelFormat128bppRGBAFloat
        || source_format==GUID_WICPixelFormat32bppRGBA || source_format==GUID_WICPixelFormat32bppBGRA;
    if(high_depth) {
        ComPtr<IWICComponentInfo> component; check(f->CreateComponentInfo(source_format,&component));
        ComPtr<IWICPixelFormatInfo> info; check(component.As(&info));
        BYTE mask[32]{}; UINT used=0; check(info->GetChannelMask(0,sizeof(mask),mask,&used));
        unsigned precision=0; for(UINT i=0;i<used;i++) for(BYTE b=mask[i];b;b>>=1) precision+=b&1;
        if(precision<=8) throw std::runtime_error("The installed Windows HEIC decoder does not expose this image's high-bit-depth precision. PaintFE has not imported an 8-bit copy.");
        if(requires_alpha && !combined_alpha) throw std::runtime_error("The installed Windows HEIC decoder cannot preserve this high-bit-depth image's transparency");
    }
    Image image; check(frame->GetSize(&image.width,&image.height));
    image.bits=high_depth?16:8;
    image.bytes.resize(pixel_bytes(image.width,image.height)*(image.bits/8));
    ComPtr<IWICFormatConverter> converter; check(f->CreateFormatConverter(&converter));
    check(converter->Initialize(frame.Get(),high_depth?GUID_WICPixelFormat64bppRGBA:GUID_WICPixelFormat32bppRGBA,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom));
    check(converter->CopyPixels(nullptr,image.width*4*(image.bits/8),static_cast<UINT>(image.bytes.size()),image.bytes.data()));
    // WIC's HEIF colour frame is RGB even when the container has auxiliary
    // alpha. Query the HEIF source transform directly; converting RGB to RGBA
    // merely synthesizes opaque alpha and silently loses the auxiliary plane.
    ComPtr<IWICBitmapSourceTransform> transform;
    bool alpha_read=combined_alpha;
    if (!combined_alpha && !high_depth && SUCCEEDED(frame.As(&transform))) {
        WICPixelFormatGUID alpha_format=GUID_WICPixelFormat8bppAlpha;
        if (SUCCEEDED(transform->GetClosestPixelFormat(&alpha_format)) && alpha_format==GUID_WICPixelFormat8bppAlpha) {
            std::vector<uint8_t> alpha(size_t(image.width)*image.height);
            check(transform->CopyPixels(nullptr,image.width,image.height,&alpha_format,WICBitmapTransformRotate0,image.width,static_cast<UINT>(alpha.size()),alpha.data()));
            for (size_t i=0;i<alpha.size();i++) image.bytes[i*4+3]=alpha[i];
            alpha_read=true;
        }
    }
    if (requires_alpha && !alpha_read)
        throw std::runtime_error("The installed Windows HEIC decoder cannot read this image's transparency. Convert it with a codec supporting this HEIC variant; PaintFE has not imported an opaque copy.");
    return image;
}
Image encode(const uint8_t* data, uint32_t w, uint32_t h, uint8_t quality) {
    COM init; auto f = factory();
    ComPtr<IStream> stream; stream.Attach(SHCreateMemStream(nullptr,0));
    if (!stream) throw std::bad_alloc();
    ComPtr<IWICBitmapEncoder> encoder; check(f->CreateEncoder(GUID_ContainerFormatHeif,nullptr,&encoder));
    check(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache));
    ComPtr<IWICBitmapFrameEncode> frame; ComPtr<IPropertyBag2> properties;
    check(encoder->CreateNewFrame(&frame,&properties));
    PROPBAG2 property{}; property.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
    VARIANT value; VariantInit(&value); value.vt = VT_R4; value.fltVal = quality/100.0f;
    check(properties->Write(1,&property,&value));
    check(frame->Initialize(properties.Get())); check(frame->SetSize(w,h));
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppRGBA; check(frame->SetPixelFormat(&format));
    bool alpha = false; for (size_t i=3;i<pixel_bytes(w,h);i+=4) if (data[i]!=255) { alpha=true; break; }
    if (alpha && format != GUID_WICPixelFormat32bppRGBA && format != GUID_WICPixelFormat32bppBGRA && format != GUID_WICPixelFormat64bppRGBA)
        throw std::runtime_error("The installed Windows HEIC encoder cannot preserve transparency. Save as PNG or PFE instead.");
    ComPtr<IWICBitmap> bitmap;
    check(f->CreateBitmapFromMemory(w,h,GUID_WICPixelFormat32bppRGBA,w*4,static_cast<UINT>(pixel_bytes(w,h)),const_cast<BYTE*>(data),&bitmap));
    ComPtr<IWICFormatConverter> converter; check(f->CreateFormatConverter(&converter));
    check(converter->Initialize(bitmap.Get(),format,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom));
    check(frame->WriteSource(converter.Get(),nullptr)); check(frame->Commit()); check(encoder->Commit());
    STATSTG stat{}; check(stream->Stat(&stat,STATFLAG_NONAME));
    if (stat.cbSize.QuadPart > 512*1024*1024) throw std::runtime_error("HEIC output exceeds limit");
    Image out; out.bytes.resize(static_cast<size_t>(stat.cbSize.QuadPart));
    LARGE_INTEGER zero{}; check(stream->Seek(zero,STREAM_SEEK_SET,nullptr)); ULONG read=0;
    check(stream->Read(out.bytes.data(),static_cast<ULONG>(out.bytes.size()),&read));
    if (read != out.bytes.size()) throw std::runtime_error("Incomplete HEIC output");
    return out;
}
void limits() {
    HANDLE job=CreateJobObjectW(nullptr,nullptr);
    if(!job) throw std::runtime_error("Could not create system codec memory limit");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    info.ProcessMemoryLimit=size_t(2)*1024*1024*1024;
    if(!SetInformationJobObject(job,JobObjectExtendedLimitInformation,&info,sizeof(info)) || !AssignProcessToJobObject(job,GetCurrentProcess())) {
        CloseHandle(job); throw std::runtime_error("Could not apply system codec memory limit");
    }
    // The handle remains alive until this short-lived worker exits.
}
