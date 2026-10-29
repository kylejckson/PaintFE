#include <windows.h>
#include <shobjidl.h>
#include <wrl/client.h>

// Remove only the exact configured resource. Shell-owned pins remain untouched.
extern "C" int paintfe_remove_recent_resource(const wchar_t* path) {
    using Microsoft::WRL::ComPtr;
    const HRESULT initialized = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(initialized) && initialized != RPC_E_CHANGED_MODE) return initialized;
    HRESULT result;
    {
        ComPtr<IApplicationDestinations> destinations;
        ComPtr<IShellItem> item;
        result = CoCreateInstance(CLSID_ApplicationDestinations, nullptr,
            CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&destinations));
        if (SUCCEEDED(result)) {
            PWSTR app_id = nullptr;
            if (SUCCEEDED(GetCurrentProcessExplicitAppUserModelID(&app_id))) {
                result = destinations->SetAppID(app_id);
            }
            CoTaskMemFree(app_id);
        }
        if (SUCCEEDED(result)) {
            result = SHCreateItemFromParsingName(path, nullptr, IID_PPV_ARGS(&item));
        }
        if (SUCCEEDED(result)) {
            result = destinations->RemoveDestination(item.Get());
            // Windows also reports this when the app has no recent list yet.
            if (result == HRESULT_FROM_WIN32(ERROR_NOT_FOUND)) result = S_OK;
        }
    }
    if (SUCCEEDED(initialized)) CoUninitialize();
    return result;
}
