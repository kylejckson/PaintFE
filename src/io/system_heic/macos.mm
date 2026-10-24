#include "bridge.hpp"
#include <Foundation/Foundation.h>
#include <ImageIO/ImageIO.h>
#include <CoreGraphics/CoreGraphics.h>
#include <mach/mach.h>
#include <thread>
#include <chrono>
template<class T> struct CF {
    T value; explicit CF(T v):value(v){} ~CF(){ if(value) CFRelease(value); }
    operator T() const { return value; }
};
uint8_t capabilities() {
    CF<CFArrayRef> inputs(CGImageSourceCopyTypeIdentifiers()), outputs(CGImageDestinationCopyTypeIdentifiers());
    auto has = [](CFArrayRef a) { return a && CFArrayContainsValue(a,CFRangeMake(0,CFArrayGetCount(a)),CFSTR("public.heic")); };
    return (has(inputs)?1:0) | (has(outputs)?2:0);
}
Image decode(const uint8_t* data, size_t size, bool requires_alpha, bool high_depth) {
    @autoreleasepool {
        CF<CFDataRef> bytes(CFDataCreate(kCFAllocatorDefault,data,size));
        CF<CGImageSourceRef> source(CGImageSourceCreateWithData(bytes,nullptr));
        if (!source.value) throw std::runtime_error("macOS could not read this HEIC image");
        CF<CFDictionaryRef> properties(CGImageSourceCopyPropertiesAtIndex(source,0,nullptr));
        if (!properties.value) throw std::runtime_error("Missing HEIC image properties");
        int width=0,height=0;
        auto number = [&](CFStringRef key,int* n) {
            auto p=static_cast<CFNumberRef>(CFDictionaryGetValue(properties,key));
            if (!p || CFGetTypeID(p)!=CFNumberGetTypeID() || !CFNumberGetValue(p,kCFNumberIntType,n)) throw std::runtime_error("Invalid HEIC dimensions");
        };
        number(kCGImagePropertyPixelWidth,&width); number(kCGImagePropertyPixelHeight,&height);
        pixel_bytes(width,height);
        NSDictionary* options=@{(__bridge NSString*)kCGImageSourceCreateThumbnailFromImageAlways:@YES,
            (__bridge NSString*)kCGImageSourceCreateThumbnailWithTransform:@YES,
            (__bridge NSString*)kCGImageSourceThumbnailMaxPixelSize:@(std::max(width,height))};
        CF<CGImageRef> image(CGImageSourceCreateThumbnailAtIndex(source,0,(__bridge CFDictionaryRef)options));
        if (!image.value) throw std::runtime_error("macOS HEIC decoding failed");
        if(high_depth && CGImageGetBitsPerComponent(image)<16)
            throw std::runtime_error("The installed macOS HEIC decoder does not expose this image's high-bit-depth precision. PaintFE has not imported an 8-bit copy.");
        auto alpha=CGImageGetAlphaInfo(image);
        if(requires_alpha && (alpha==kCGImageAlphaNone || alpha==kCGImageAlphaNoneSkipFirst || alpha==kCGImageAlphaNoneSkipLast))
            throw std::runtime_error("The installed macOS HEIC decoder cannot preserve this image's transparency");
        Image result; result.width=CGImageGetWidth(image); result.height=CGImageGetHeight(image);
        result.bits=high_depth?16:8;
        result.bytes.resize(pixel_bytes(result.width,result.height)*(result.bits/8));
        CF<CGColorSpaceRef> color(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
        CF<CGContextRef> context(CGBitmapContextCreate(result.bytes.data(),result.width,result.height,result.bits,result.width*4*(result.bits/8),color,kCGImageAlphaPremultipliedLast|(high_depth?kCGBitmapByteOrder16Little:kCGBitmapByteOrder32Big)));
        if (!context.value) throw std::runtime_error("macOS HEIC raster allocation failed");
        CGContextDrawImage(context,CGRectMake(0,0,result.width,result.height),image);
        if(high_depth) {
            auto sample=[&](size_t i)->uint32_t{return result.bytes[i]|(uint32_t(result.bytes[i+1])<<8);};
            for(size_t i=0;i<result.bytes.size();i+=8) {
                auto a=sample(i+6);
                for(size_t c=0;c<6;c+=2) {
                    uint32_t value=a?std::min(65535u,(sample(i+c)*65535+a/2)/a):0;
                    result.bytes[i+c]=value&255; result.bytes[i+c+1]=value>>8;
                }
            }
        } else for (size_t i=0;i<result.bytes.size();i+=4) {
            auto a=result.bytes[i+3];
            for (size_t c=0;c<3;c++) result.bytes[i+c]=a?std::min(255u,(unsigned(result.bytes[i+c])*255+a/2)/a):0;
        }
        return result;
    }
}
Image encode(const uint8_t* data, uint32_t w, uint32_t h, uint8_t quality) {
    @autoreleasepool {
        CF<CFDataRef> bytes(CFDataCreate(kCFAllocatorDefault,data,pixel_bytes(w,h)));
        CF<CGDataProviderRef> provider(CGDataProviderCreateWithCFData(bytes));
        CF<CGColorSpaceRef> color(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
        CF<CGImageRef> image(CGImageCreate(w,h,8,32,w*4,color,CGBitmapInfo(kCGImageAlphaLast|kCGBitmapByteOrder32Big),provider,nullptr,false,kCGRenderingIntentDefault));
        CF<CFMutableDataRef> output(CFDataCreateMutable(kCFAllocatorDefault,0));
        CF<CGImageDestinationRef> destination(CGImageDestinationCreateWithData(output,CFSTR("public.heic"),1,nullptr));
        if (!image.value || !destination.value) throw std::runtime_error("HEIC encoding is unavailable on this macOS installation");
        NSDictionary* options=@{(__bridge NSString*)kCGImageDestinationLossyCompressionQuality:@(quality/100.0)};
        CGImageDestinationAddImage(destination,image,(__bridge CFDictionaryRef)options);
        if (!CGImageDestinationFinalize(destination)) throw std::runtime_error("macOS HEIC encoding failed. Try PNG or PFE.");
        Image result; auto length=CFDataGetLength(output);
        if (length>512*1024*1024) throw std::runtime_error("HEIC output exceeds limit");
        result.bytes.assign(CFDataGetBytePtr(output),CFDataGetBytePtr(output)+length);
        return result;
    }
}
void limits() {
    std::thread([] {
        for(;;) {
            mach_task_basic_info_data_t info{}; mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
            if(task_info(mach_task_self(),MACH_TASK_BASIC_INFO,reinterpret_cast<task_info_t>(&info),&count)!=KERN_SUCCESS || info.resident_size>uint64_t(2)*1024*1024*1024) std::_Exit(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }).detach();
}
