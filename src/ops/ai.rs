// ============================================================================
// AI OPERATIONS — ONNX Runtime integration for neural network inference
// ============================================================================
//
// Uses `libloading` to dynamically load onnxruntime.dll / libonnxruntime.so
// at runtime so the binary has NO compile-time dependency on ONNX Runtime.
// The user configures the DLL path in Settings → AI.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_unsafe)]

use image::RgbaImage;
use std::path::Path;

/// Settings for the Remove Background operation.
/// These control how the AI-generated mask is post-processed before
/// being applied as the alpha channel.
#[derive(Clone, Debug)]
pub struct RemoveBgSettings {
    /// Foreground probability threshold (0.0–1.0). Pixels with probability
    /// below this are considered background. Higher values = keep more
    /// (more conservative). Default: 0.5.
    pub threshold: f32,
    /// Edge feathering radius in pixels. Applies a Gaussian blur to the
    /// mask to soften hard edges. 0 = no feathering. Default: 0.0.
    pub edge_feather: f32,
    /// Mask expansion in pixels. Positive = grow foreground (keep more),
    /// negative = shrink foreground (remove more). Default: 0.
    pub mask_expansion: i32,
    /// When true, use smooth alpha transitions around the threshold
    /// instead of a hard cutoff. Default: true.
    pub smooth_edges: bool,
    /// Fill interior holes in the foreground mask using morphological close
    /// (dilate then erode). The value is the kernel radius. 0 = disabled.
    /// Default: 0.
    pub fill_holes: u32,
}

impl Default for RemoveBgSettings {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            edge_feather: 0.0,
            mask_expansion: 0,
            smooth_edges: true,
            fill_holes: 0,
        }
    }
}

/// Errors that can occur during ONNX Runtime operations.
#[derive(Debug)]
pub enum OnnxError {
    DllNotFound(String),
    DllLoadFailed(String),
    ModelNotFound(String),
    ModelLoadFailed(String),
    ApiInitFailed(String),
    SessionCreateFailed(String),
    InferenceFailed(String),
    InvalidOutput(String),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::DllNotFound(p) => write!(f, "ONNX Runtime DLL not found: {}", p),
            OnnxError::DllLoadFailed(e) => write!(f, "Failed to load ONNX Runtime DLL: {}", e),
            OnnxError::ModelNotFound(p) => write!(f, "ONNX model file not found: {}", p),
            OnnxError::ModelLoadFailed(e) => write!(f, "Failed to load ONNX model: {}", e),
            OnnxError::ApiInitFailed(e) => write!(f, "ONNX Runtime API init failed: {}", e),
            OnnxError::SessionCreateFailed(e) => write!(f, "Failed to create ONNX session: {}", e),
            OnnxError::InferenceFailed(e) => write!(f, "ONNX inference failed: {}", e),
            OnnxError::InvalidOutput(e) => write!(f, "Invalid ONNX output: {}", e),
        }
    }
}

// --- ONNX Runtime C API types --------------------------------------
// These mirror the C API structs from onnxruntime_c_api.h.
// We only define the opaque handle types and the function pointer tables
// we actually need.

/// Opaque handles (never dereferenced in Rust — used as `*mut` pointers only)
#[repr(C)]
struct OrtEnv {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtSession {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtSessionOptions {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtValue {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtMemoryInfo {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtStatus {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtRunOptions {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtAllocator {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtTensorTypeAndShapeInfo {
    _private: [u8; 0],
}
#[repr(C)]
struct OrtTypeInfo {
    _private: [u8; 0],
}

/// ORT API version we target (compatible with ONNX Runtime 1.14+)
const ORT_API_VERSION: u32 = 18;

/// Logging level
#[allow(dead_code)]
#[repr(u32)]
enum OrtLoggingLevel {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4,
}

/// Tensor element type
#[allow(dead_code)]
#[repr(u32)]
enum ONNXTensorElementDataType {
    Undefined = 0,
    Float = 1,
    UInt8 = 2,
    Int8 = 3,
    UInt16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    UInt32 = 12,
    UInt64 = 13,
}

/// Allocator type
#[repr(i32)]
#[allow(dead_code)]
enum OrtAllocatorType {
    Invalid = -1,
    DeviceAllocator = 0,
    ArenaAllocator = 1,
}

/// Memory type
#[repr(i32)]
#[allow(dead_code)]
enum OrtMemType {
    CpuInput = -2,
    CpuOutput = -1,
    Default = 0,
}

/// The subset of the OrtApi vtable we actually use.
/// The real struct has ~200 function pointers; we index into it by field offset.
///
/// OrtApi is a struct of function pointers. We load the whole blob and cast
/// to get individual function pointers by their index (each pointer is 8 bytes
/// on 64-bit, 4 bytes on 32-bit).
struct OrtApi {
    raw: *const std::ffi::c_void,
}

// Function pointer type aliases (C calling convention)
type CreateEnvFn = unsafe extern "C" fn(
    log_level: OrtLoggingLevel,
    logid: *const std::ffi::c_char,
    out: *mut *mut OrtEnv,
) -> *mut OrtStatus;

type CreateSessionOptionsFn =
    unsafe extern "C" fn(out: *mut *mut OrtSessionOptions) -> *mut OrtStatus;

type CreateSessionFn = unsafe extern "C" fn(
    env: *const OrtEnv,
    model_path: *const u16, // Wide string on Windows
    options: *const OrtSessionOptions,
    out: *mut *mut OrtSession,
) -> *mut OrtStatus;

type CreateTensorWithDataAsOrtValueFn = unsafe extern "C" fn(
    info: *const OrtMemoryInfo,
    data: *mut std::ffi::c_void,
    data_len: usize,
    shape: *const i64,
    shape_len: usize,
    element_type: ONNXTensorElementDataType,
    out: *mut *mut OrtValue,
) -> *mut OrtStatus;

type CreateCpuMemoryInfoFn = unsafe extern "C" fn(
    alloc_type: OrtAllocatorType,
    mem_type: OrtMemType,
    out: *mut *mut OrtMemoryInfo,
) -> *mut OrtStatus;

type RunFn = unsafe extern "C" fn(
    session: *mut OrtSession,
    run_options: *const OrtRunOptions,
    input_names: *const *const std::ffi::c_char,
    inputs: *const *const OrtValue,
    input_count: usize,
    output_names: *const *const std::ffi::c_char,
    output_count: usize,
    outputs: *mut *mut OrtValue,
) -> *mut OrtStatus;

type GetTensorMutableDataFn =
    unsafe extern "C" fn(value: *mut OrtValue, out: *mut *mut std::ffi::c_void) -> *mut OrtStatus;

type GetTensorTypeAndShapeFn = unsafe extern "C" fn(
    value: *const OrtValue,
    out: *mut *mut OrtTensorTypeAndShapeInfo,
) -> *mut OrtStatus;

type GetDimensionsCountFn =
    unsafe extern "C" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> *mut OrtStatus;

type GetDimensionsFn = unsafe extern "C" fn(
    info: *const OrtTensorTypeAndShapeInfo,
    dim_values: *mut i64,
    dim_values_length: usize,
) -> *mut OrtStatus;

type ReleaseEnvFn = unsafe extern "C" fn(env: *mut OrtEnv);
type ReleaseSessionFn = unsafe extern "C" fn(session: *mut OrtSession);
type ReleaseSessionOptionsFn = unsafe extern "C" fn(options: *mut OrtSessionOptions);
type ReleaseValueFn = unsafe extern "C" fn(value: *mut OrtValue);
type ReleaseMemoryInfoFn = unsafe extern "C" fn(info: *mut OrtMemoryInfo);
type ReleaseTensorTypeAndShapeInfoFn = unsafe extern "C" fn(info: *mut OrtTensorTypeAndShapeInfo);
type ReleaseStatusFn = unsafe extern "C" fn(status: *mut OrtStatus);
type GetErrorMessageFn = unsafe extern "C" fn(status: *const OrtStatus) -> *const std::ffi::c_char;
type SetIntraOpNumThreadsFn = unsafe extern "C" fn(
    options: *mut OrtSessionOptions,
    intra_op_num_threads: i32,
) -> *mut OrtStatus;
type SetSessionGraphOptimizationLevelFn = unsafe extern "C" fn(
    options: *mut OrtSessionOptions,
    graph_optimization_level: u32,
) -> *mut OrtStatus;

type SessionGetInputCountFn =
    unsafe extern "C" fn(session: *const OrtSession, out: *mut usize) -> *mut OrtStatus;

type SessionGetOutputCountFn =
    unsafe extern "C" fn(session: *const OrtSession, out: *mut usize) -> *mut OrtStatus;

type SessionGetInputNameFn = unsafe extern "C" fn(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
    out: *mut *mut std::ffi::c_char,
) -> *mut OrtStatus;

type SessionGetOutputNameFn = unsafe extern "C" fn(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
    out: *mut *mut std::ffi::c_char,
) -> *mut OrtStatus;

type GetAllocatorWithDefaultOptionsFn =
    unsafe extern "C" fn(out: *mut *mut OrtAllocator) -> *mut OrtStatus;

type AllocatorFreeFn = unsafe extern "C" fn(
    allocator: *mut OrtAllocator,
    ptr: *mut std::ffi::c_void,
) -> *mut OrtStatus;

type SessionGetInputTypeInfoFn = unsafe extern "C" fn(
    session: *const OrtSession,
    index: usize,
    type_info: *mut *mut OrtTypeInfo,
) -> *mut OrtStatus;

type CastTypeInfoToTensorInfoFn = unsafe extern "C" fn(
    type_info: *const OrtTypeInfo,
    out: *mut *const OrtTensorTypeAndShapeInfo,
) -> *mut OrtStatus;

type ReleaseTypeInfoFn = unsafe extern "C" fn(type_info: *mut OrtTypeInfo);

/// OrtApiBase — the entry point struct returned by OrtGetApiBase()
#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "C" fn(version: u32) -> *const std::ffi::c_void,
    get_version_string: unsafe extern "C" fn() -> *const std::ffi::c_char,
}

impl OrtApi {
    /// Get a function pointer from the API vtable by index.
    /// The OrtApi struct is an array of function pointers.
    unsafe fn get_fn<T>(&self, index: usize) -> T {
        let ptr = self.raw as *const *const std::ffi::c_void;
        let fn_ptr = *ptr.add(index);
        std::mem::transmute_copy(&fn_ptr)
    }

    // OrtApi function indices (from onnxruntime_c_api.h)
    // These indices correspond to the position of each function pointer
    // in the OrtApi struct. Carefully counted from the official header.
    //
    // Struct layout (first 102 entries):
    //  0: CreateStatus            1: GetErrorCode           2: GetErrorMessage
    //  3: CreateEnv               4: CreateEnvWithCustomLogger
    //  5: EnableTelemetryEvents   6: DisableTelemetryEvents
    //  7: CreateSession           8: CreateSessionFromArray  9: Run
    // 10: CreateSessionOptions   11: SetOptimizedModelFilePath
    // 12: CloneSessionOptions    13: SetSessionExecutionMode
    // 14: EnableProfiling        15: DisableProfiling
    // 16: EnableMemPattern       17: DisableMemPattern
    // 18: EnableCpuMemArena      19: DisableCpuMemArena
    // 20: SetSessionLogId        21: SetSessionLogVerbosityLevel
    // 22: SetSessionLogSeverityLevel
    // 23: SetSessionGraphOptimizationLevel
    // 24: SetIntraOpNumThreads   25: SetInterOpNumThreads
    // 26: CreateCustomOpDomain   27: CustomOpDomain_Add
    // 28: AddCustomOpDomain      29: RegisterCustomOpsLibrary
    // 30: SessionGetInputCount   31: SessionGetOutputCount
    // 32: SessionGetOverridableInitializerCount
    // 33: SessionGetInputTypeInfo 34: SessionGetOutputTypeInfo
    // 35: SessionGetOverridableInitializerTypeInfo
    // 36: SessionGetInputName    37: SessionGetOutputName
    // 38: SessionGetOverridableInitializerName
    // 39: CreateRunOptions
    // 40-47: RunOptions get/set functions
    // 48: CreateTensorAsOrtValue 49: CreateTensorWithDataAsOrtValue
    // 50: IsTensor               51: GetTensorMutableData
    // 52-54: String tensor functions
    // 55: CastTypeInfoToTensorInfo 56: GetOnnxTypeFromTypeInfo
    // 57: CreateTensorTypeAndShapeInfo
    // 58: SetTensorElementType   59: SetDimensions
    // 60: GetTensorElementType   61: GetDimensionsCount
    // 62: GetDimensions          63: GetSymbolicDimensions
    // 64: GetTensorShapeElementCount
    // 65: GetTensorTypeAndShape  66: GetTypeInfo  67: GetValueType
    // 68: CreateMemoryInfo       69: CreateCpuMemoryInfo
    // 70: CompareMemoryInfo      71-74: MemoryInfo getters
    // 75: AllocatorAlloc         76: AllocatorFree  77: AllocatorGetInfo
    // 78: GetAllocatorWithDefaultOptions
    // 79: AddFreeDimensionOverride
    // 80-84: GetValue, GetValueCount, CreateValue, CreateOpaqueValue, GetOpaqueValue
    // 85-87: KernelInfoGetAttribute_float/int64/string
    // 88-91: KernelContext_GetInputCount/OutputCount/Input/Output
    // 92: ReleaseEnv             93: ReleaseStatus
    // 94: ReleaseMemoryInfo      95: ReleaseSession
    // 96: ReleaseValue           97: ReleaseRunOptions
    // 98: ReleaseTypeInfo        99: ReleaseTensorTypeAndShapeInfo
    // 100: ReleaseSessionOptions 101: ReleaseCustomOpDomain

    fn create_env(&self) -> CreateEnvFn {
        unsafe { self.get_fn(3) }
    }
    fn create_session_options(&self) -> CreateSessionOptionsFn {
        unsafe { self.get_fn(10) }
    }
    fn create_session(&self) -> CreateSessionFn {
        unsafe { self.get_fn(7) }
    }
    fn run(&self) -> RunFn {
        unsafe { self.get_fn(9) }
    }
    fn create_tensor_with_data(&self) -> CreateTensorWithDataAsOrtValueFn {
        unsafe { self.get_fn(49) }
    }
    fn create_cpu_memory_info(&self) -> CreateCpuMemoryInfoFn {
        unsafe { self.get_fn(69) }
    }
    fn get_tensor_mutable_data(&self) -> GetTensorMutableDataFn {
        unsafe { self.get_fn(51) }
    }
    fn get_tensor_type_and_shape(&self) -> GetTensorTypeAndShapeFn {
        unsafe { self.get_fn(65) }
    }
    fn get_dimensions_count(&self) -> GetDimensionsCountFn {
        unsafe { self.get_fn(61) }
    }
    fn get_dimensions(&self) -> GetDimensionsFn {
        unsafe { self.get_fn(62) }
    }
    fn release_env(&self) -> ReleaseEnvFn {
        unsafe { self.get_fn(92) }
    }
    fn release_session(&self) -> ReleaseSessionFn {
        unsafe { self.get_fn(95) }
    }
    fn release_session_options(&self) -> ReleaseSessionOptionsFn {
        unsafe { self.get_fn(100) }
    }
    fn release_value(&self) -> ReleaseValueFn {
        unsafe { self.get_fn(96) }
    }
    fn release_memory_info(&self) -> ReleaseMemoryInfoFn {
        unsafe { self.get_fn(94) }
    }
    fn release_tensor_type_and_shape_info(&self) -> ReleaseTensorTypeAndShapeInfoFn {
        unsafe { self.get_fn(99) }
    }
    fn release_status(&self) -> ReleaseStatusFn {
        unsafe { self.get_fn(93) }
    }
    fn get_error_message(&self) -> GetErrorMessageFn {
        unsafe { self.get_fn(2) }
    }
    fn set_intra_op_num_threads(&self) -> SetIntraOpNumThreadsFn {
        unsafe { self.get_fn(24) }
    }
    fn set_session_graph_optimization_level(&self) -> SetSessionGraphOptimizationLevelFn {
        unsafe { self.get_fn(23) }
    }
    fn session_get_input_count(&self) -> SessionGetInputCountFn {
        unsafe { self.get_fn(30) }
    }
    fn session_get_output_count(&self) -> SessionGetOutputCountFn {
        unsafe { self.get_fn(31) }
    }
    fn session_get_input_name(&self) -> SessionGetInputNameFn {
        unsafe { self.get_fn(36) }
    }
    fn session_get_output_name(&self) -> SessionGetOutputNameFn {
        unsafe { self.get_fn(37) }
    }
    fn get_allocator_with_default_options(&self) -> GetAllocatorWithDefaultOptionsFn {
        unsafe { self.get_fn(78) }
    }
    fn allocator_free(&self) -> AllocatorFreeFn {
        unsafe { self.get_fn(76) }
    }
    fn session_get_input_type_info(&self) -> SessionGetInputTypeInfoFn {
        unsafe { self.get_fn(33) }
    }
    fn cast_type_info_to_tensor_info(&self) -> CastTypeInfoToTensorInfoFn {
        unsafe { self.get_fn(55) }
    }
    fn release_type_info(&self) -> ReleaseTypeInfoFn {
        unsafe { self.get_fn(98) }
    }
}

/// Extract error message from an OrtStatus pointer. Returns None if status is null (success).
unsafe fn status_to_result(api: &OrtApi, status: *mut OrtStatus) -> Result<(), String> {
    if status.is_null() {
        Ok(())
    } else {
        let msg_ptr = (api.get_error_message())(status);
        let msg = if msg_ptr.is_null() {
            "Unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(msg_ptr)
                .to_string_lossy()
                .into_owned()
        };
        (api.release_status())(status);
        Err(msg)
    }
}

/// Minimum supported ONNX Runtime version (1.16.0).
/// Versions older than this used a different vtable layout for API version 18.
const ORT_MIN_VERSION: (u32, u32) = (1, 16);

/// Validate that an ONNX DLL/model path is safe to load:
/// - Must be an absolute path (no relative traversal)
/// - Must not contain `..` components (path traversal guard)
/// - Must have the correct file extension for the expected type
pub fn validate_onnx_path(path: &str, for_dll: bool) -> Result<(), OnnxError> {
    use std::path::Component;
    let p = Path::new(path);

    // Reject empty path
    if path.is_empty() {
        return Err(OnnxError::DllNotFound("Path is empty".to_string()));
    }

    // Must be absolute — prevents loading from relative/CWD-based paths
    if !p.is_absolute() {
        return Err(OnnxError::DllLoadFailed(
            "ONNX path must be an absolute path (e.g. C:\\...\\onnxruntime.dll)".to_string(),
        ));
    }

    // Reject any path containing `..` traversal components
    for component in p.components() {
        if component == Component::ParentDir {
            return Err(OnnxError::DllLoadFailed(
                "ONNX path must not contain '..' components".to_string(),
            ));
        }
    }

    // Check extension matches expected type
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if for_dll {
        let valid_dll_exts = ["dll", "so", "dylib"];
        if !valid_dll_exts.contains(&ext.as_str()) {
            return Err(OnnxError::DllLoadFailed(format!(
                "Expected a .dll/.so/.dylib file, got '.{}'",
                ext
            )));
        }
    } else if ext != "onnx" {
        return Err(OnnxError::ModelLoadFailed(format!(
            "Expected a .onnx model file, got '.{}'",
            ext
        )));
    }

    Ok(())
}

/// Parse a version string like "1.18.0" into (major, minor) tuple.
fn parse_ort_version(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.split('.');
    let major: u32 = parts.next()?.trim().parse().ok()?;
    let minor: u32 = parts.next()?.trim().parse().ok()?;
    Some((major, minor))
}

/// Probe the ONNX Runtime DLL. Returns the version string on success; rejects unsafe paths and enforces >= 1.16.
pub fn probe_onnx_runtime(dll_path: &str) -> Result<String, OnnxError> {
    // Security: validate path before loading
    validate_onnx_path(dll_path, true)?;

    if !Path::new(dll_path).exists() {
        return Err(OnnxError::DllNotFound(dll_path.to_string()));
    }

    unsafe {
        let lib = libloading::Library::new(dll_path)
            .map_err(|e| OnnxError::DllLoadFailed(format!("{}", e)))?;

        let get_api_base: libloading::Symbol<unsafe extern "C" fn() -> *const OrtApiBase> =
            lib.get(b"OrtGetApiBase").map_err(|e| {
                OnnxError::DllLoadFailed(format!("Symbol OrtGetApiBase not found: {}", e))
            })?;

        let api_base = get_api_base();
        if api_base.is_null() {
            return Err(OnnxError::ApiInitFailed(
                "OrtGetApiBase returned null".to_string(),
            ));
        }

        // Get version string
        let version_ptr = ((*api_base).get_version_string)();
        let version = if version_ptr.is_null() {
            "unknown".to_string()
        } else {
            std::ffi::CStr::from_ptr(version_ptr)
                .to_string_lossy()
                .into_owned()
        };

        // Enforce minimum version — older builds have incompatible vtable layouts
        if let Some((major, minor)) = parse_ort_version(&version) {
            let (min_major, min_minor) = ORT_MIN_VERSION;
            let too_old = major < min_major || (major == min_major && minor < min_minor);
            if too_old {
                return Err(OnnxError::ApiInitFailed(format!(
                    "ONNX Runtime {} is too old. Minimum supported version is {}.{}. \
                     Please download a newer release from https://github.com/microsoft/onnxruntime/releases",
                    version, min_major, min_minor
                )));
            }
        }

        // Verify we can get the API
        let api_ptr = ((*api_base).get_api)(ORT_API_VERSION);
        if api_ptr.is_null() {
            return Err(OnnxError::ApiInitFailed(format!(
                "OrtGetApi({}) returned null — DLL version {} may be too old",
                ORT_API_VERSION, version
            )));
        }

        Ok(version)
    }
}

/// Default fallback input dimensions (BiRefNet default)
const DEFAULT_MODEL_SIZE: u32 = 1024;

/// Detected model profile for background removal.
/// Auto-detected from model metadata (input shape + output count).
#[derive(Debug, Clone, Copy)]
enum ModelProfile {
    /// BiRefNet: 1024×1024 input, best output = last (most refined decoder stage)
    BiRefNet,
    /// U²-Net: 320×320 input, best output = first (d0 is the refined output)
    U2Net,
    /// IS-Net: 1024×1024 input, best output = first
    ISNet,
    /// Unknown model — best output chosen by confidence scoring
    Unknown,
}

impl ModelProfile {
    /// Detect model profile from input dimensions and output count.
    fn detect(input_h: u32, input_w: u32, output_count: usize) -> Self {
        if input_h == 320 && input_w == 320 {
            ModelProfile::U2Net
        } else if input_h == 1024 && input_w == 1024 {
            // Both BiRefNet and IS-Net use 1024×1024.
            // BiRefNet typically has 5+ decoder outputs; IS-Net / simpler models have fewer.
            if output_count >= 5 {
                ModelProfile::BiRefNet
            } else {
                ModelProfile::ISNet
            }
        } else {
            ModelProfile::Unknown
        }
    }

    fn description(&self) -> &'static str {
        match self {
            ModelProfile::BiRefNet => "BiRefNet (1024x1024, prefer last output)",
            ModelProfile::U2Net => "U²-Net (320x320, prefer first output)",
            ModelProfile::ISNet => "IS-Net (1024x1024, prefer first output)",
            ModelProfile::Unknown => "Unknown model (auto-selecting best output)",
        }
    }

    /// Returns the preferred output index for this model profile, given the total output count.
    /// Used as a tiebreaker when confidence scores are close.
    fn preferred_output_index(&self, output_count: usize) -> usize {
        match self {
            ModelProfile::BiRefNet => output_count.saturating_sub(1), // last
            ModelProfile::U2Net => 0,                                 // first (d0)
            ModelProfile::ISNet => 0,                                 // first
            ModelProfile::Unknown => 0,                               // default to first
        }
    }
}

/// Detect whether model output values are already probabilities (in [0, 1])
/// or raw logits (arbitrary range requiring sigmoid).
/// U²-Net and IS-Net output sigmoid-activated probabilities;
/// BiRefNet outputs raw logits.
fn is_probability_space(data: &[f32]) -> bool {
    if data.is_empty() {
        return false;
    }
    // Sample up to 10000 evenly-spaced values for speed
    let step = (data.len() / 10000).max(1);
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for i in (0..data.len()).step_by(step) {
        min_val = min_val.min(data[i]);
        max_val = max_val.max(data[i]);
    }
    // If all values in [-0.05, 1.05], it's probability space
    min_val >= -0.05 && max_val <= 1.05
}

/// Convert a raw value to probability, applying sigmoid only if needed.
#[inline]
fn to_probability(v: f32, already_prob: bool) -> f32 {
    if already_prob {
        v.clamp(0.0, 1.0)
    } else {
        1.0 / (1.0 + (-v).exp())
    }
}

/// Score a mask output by how "confident" / decisive its predictions are.
/// Returns the fraction of pixels with probability > 0.9 or < 0.1.
/// A more refined/final decoder stage will have a higher score because
/// its predictions are more bimodal (closer to 0 or 1).
/// Auto-detects whether values are logits or probabilities.
pub fn mask_confidence_score(data: &[f32]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let is_prob = is_probability_space(data);
    let decisive: usize = data
        .iter()
        .filter(|&&v| {
            let prob = to_probability(v, is_prob) as f64;
            !(0.1..=0.9).contains(&prob)
        })
        .count();
    decisive as f64 / data.len() as f64
}

/// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess an RGBA image for segmentation model inference:
/// 1. Resize to target_size × target_size
/// 2. Normalize with ImageNet mean/std
/// 3. Convert to CHW float32 layout
///
/// Returns a Vec<f32> in [1, 3, target_size, target_size] layout.
fn preprocess_image(input: &RgbaImage, target_size: u32) -> Vec<f32> {
    let resized = image::imageops::resize(
        input,
        target_size,
        target_size,
        image::imageops::FilterType::Lanczos3,
    );

    let npixels = (target_size * target_size) as usize;
    let mut tensor = vec![0.0f32; 3 * npixels];

    for y in 0..target_size {
        for x in 0..target_size {
            let pixel = resized.get_pixel(x, y);
            let idx = (y * target_size + x) as usize;
            // Channel-first layout: [R-plane, G-plane, B-plane]
            tensor[idx] = (pixel[0] as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            tensor[npixels + idx] = (pixel[1] as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            tensor[2 * npixels + idx] =
                (pixel[2] as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
    }

    tensor
}

/// Postprocess model output probabilities and apply to original image:
/// 1. Apply threshold / smooth edge processing to probabilities
/// 2. Optionally expand/contract mask
/// 3. Optionally feather (blur) mask edges
/// 4. Resize mask to original dimensions
/// 5. Apply mask as alpha channel
///
/// `mask_probs` must contain foreground probabilities in [0.0, 1.0].
/// Caller is responsible for converting logits → probabilities first.
fn postprocess_mask(
    mask_probs: &[f32],
    mask_h: u32,
    mask_w: u32,
    original: &RgbaImage,
    settings: &RemoveBgSettings,
) -> RgbaImage {
    let (orig_w, orig_h) = original.dimensions();

    // Apply threshold to probabilities
    let mask_pixels: Vec<u8> = mask_probs
        .iter()
        .map(|&prob| {
            if settings.smooth_edges {
                // Smooth transition: remap probabilities around threshold
                // using a steep sigmoid-like curve centered at threshold
                let steepness = 12.0; // Controls how sharp the transition is
                let remapped = 1.0 / (1.0 + (-(prob - settings.threshold) * steepness).exp());
                (remapped * 255.0).clamp(0.0, 255.0) as u8
            } else {
                // Hard cutoff at threshold
                if prob >= settings.threshold {
                    255u8
                } else {
                    0u8
                }
            }
        })
        .collect();

    // Create a grayscale mask image
    let mut mask_img = image::GrayImage::from_raw(mask_w, mask_h, mask_pixels)
        .unwrap_or_else(|| image::GrayImage::new(mask_w, mask_h));

    // Apply mask expansion/contraction (dilate/erode) at model resolution
    if settings.mask_expansion != 0 {
        mask_img = apply_mask_expansion(&mask_img, settings.mask_expansion);
    }

    // Fill interior holes via morphological close (dilate then erode)
    // This fills small gaps/holes inside the foreground region
    if settings.fill_holes > 0 {
        eprintln!(
            "[AI]   Applying fill holes (radius {})",
            settings.fill_holes
        );
        mask_img = morphological_close(&mask_img, settings.fill_holes as i32);
    }

    // Resize mask to original dimensions
    let mut resized_mask = if mask_w != orig_w || mask_h != orig_h {
        image::imageops::resize(
            &mask_img,
            orig_w,
            orig_h,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        mask_img
    };

    // Apply edge feathering (Gaussian blur on mask) at full resolution
    if settings.edge_feather > 0.5 {
        resized_mask = blur_grayscale(&resized_mask, settings.edge_feather);
    }

    // Apply mask as alpha channel
    let mut output = original.clone();
    for y in 0..orig_h {
        for x in 0..orig_w {
            let mask_val = resized_mask.get_pixel(x, y)[0];
            let pixel = output.get_pixel_mut(x, y);
            // Combine existing alpha with mask
            let orig_alpha = pixel[3] as f32 / 255.0;
            let mask_alpha = mask_val as f32 / 255.0;
            pixel[3] = (orig_alpha * mask_alpha * 255.0).clamp(0.0, 255.0) as u8;
        }
    }

    output
}

/// Apply mask expansion (positive = dilate/grow foreground) or
/// contraction (negative = erode/shrink foreground).
fn apply_mask_expansion(mask: &image::GrayImage, expansion: i32) -> image::GrayImage {
    let (w, h) = mask.dimensions();
    let iterations = expansion.unsigned_abs() as usize;
    let mut current = mask.clone();

    for _ in 0..iterations {
        let mut next = current.clone();
        for y in 0..h {
            for x in 0..w {
                let center = current.get_pixel(x, y)[0];
                if expansion > 0 {
                    // Dilate: if any neighbor is foreground, become foreground
                    if center < 128 {
                        let mut max_val = center;
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                                    max_val =
                                        max_val.max(current.get_pixel(nx as u32, ny as u32)[0]);
                                }
                            }
                        }
                        next.put_pixel(x, y, image::Luma([max_val]));
                    }
                } else {
                    // Erode: if any neighbor is background, become background
                    if center > 128 {
                        let mut min_val = center;
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                                    min_val =
                                        min_val.min(current.get_pixel(nx as u32, ny as u32)[0]);
                                }
                            }
                        }
                        next.put_pixel(x, y, image::Luma([min_val]));
                    }
                }
            }
        }
        current = next;
    }

    current
}

/// Morphological close operation: dilate then erode.
/// This fills interior holes/gaps in the foreground mask without
/// expanding the outer boundary (the erode step restores it).
fn morphological_close(mask: &image::GrayImage, radius: i32) -> image::GrayImage {
    // Step 1: Dilate (expand foreground to fill holes)
    let dilated = apply_mask_expansion(mask, radius);
    // Step 2: Erode (shrink back to original boundary)
    apply_mask_expansion(&dilated, -radius)
}

/// Simple box-approximated Gaussian blur for grayscale images.
/// Uses separable horizontal + vertical passes.
fn blur_grayscale(img: &image::GrayImage, radius: f32) -> image::GrayImage {
    let (w, h) = img.dimensions();
    let r = (radius.ceil() as i32).max(1);
    let _kernel_size = (2 * r + 1) as f32;

    // Horizontal pass
    let mut temp = image::GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            for dx in -r..=r {
                let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                sum += img.get_pixel(nx, y)[0] as f32;
                count += 1.0;
            }
            temp.put_pixel(x, y, image::Luma([(sum / count) as u8]));
        }
    }

    // Vertical pass
    let mut result = image::GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            for dy in -r..=r {
                let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                sum += temp.get_pixel(x, ny)[0] as f32;
                count += 1.0;
            }
            result.put_pixel(x, y, image::Luma([(sum / count) as u8]));
        }
    }

    result
}

/// Remove the background from an image using a segmentation model via ONNX Runtime.
///
/// Supports multiple model architectures (BiRefNet, U²-Net, IS-Net) with
/// automatic detection based on model input shape and output structure.
///
/// This function:
/// 1. Dynamically loads onnxruntime.dll
/// 2. Creates an inference session with the segmentation model
/// 3. Auto-detects model type from input dimensions and output count
/// 4. Preprocesses the input (resize to model size, normalize, CHW layout)
/// 5. Runs inference and auto-selects the best output by confidence scoring
/// 6. Applies the mask as alpha to the original image (with settings)
///
/// Returns the image with background made transparent.
pub fn remove_background(
    dll_path: &str,
    model_path: &str,
    input: &RgbaImage,
    settings: &RemoveBgSettings,
) -> Result<RgbaImage, OnnxError> {
    eprintln!("[AI] remove_background: starting");
    eprintln!("[AI]   dll_path:   {}", dll_path);
    eprintln!("[AI]   model_path: {}", model_path);
    eprintln!("[AI]   input size: {}x{}", input.width(), input.height());

    // Security: validate paths before loading any native code
    validate_onnx_path(dll_path, true)?;
    validate_onnx_path(model_path, false)?;

    if !Path::new(dll_path).exists() {
        return Err(OnnxError::DllNotFound(dll_path.to_string()));
    }
    if !Path::new(model_path).exists() {
        return Err(OnnxError::ModelNotFound(model_path.to_string()));
    }

    unsafe {
        // -- Load library --
        eprintln!("[AI] Loading ONNX Runtime DLL...");
        let lib = libloading::Library::new(dll_path)
            .map_err(|e| OnnxError::DllLoadFailed(format!("{}", e)))?;

        let get_api_base: libloading::Symbol<unsafe extern "C" fn() -> *const OrtApiBase> = lib
            .get(b"OrtGetApiBase")
            .map_err(|e| OnnxError::DllLoadFailed(format!("Symbol not found: {}", e)))?;

        let api_base = get_api_base();
        if api_base.is_null() {
            return Err(OnnxError::ApiInitFailed(
                "OrtGetApiBase returned null".to_string(),
            ));
        }

        let api_ptr = ((*api_base).get_api)(ORT_API_VERSION);
        if api_ptr.is_null() {
            return Err(OnnxError::ApiInitFailed(format!(
                "OrtGetApi({}) returned null",
                ORT_API_VERSION
            )));
        }
        let api = OrtApi { raw: api_ptr };
        eprintln!("[AI] OrtApi loaded successfully");

        // -- Create environment --
        eprintln!("[AI] Creating environment...");
        let mut env: *mut OrtEnv = std::ptr::null_mut();
        let log_id = std::ffi::CString::new("PaintFE").unwrap();
        status_to_result(
            &api,
            (api.create_env())(OrtLoggingLevel::Warning, log_id.as_ptr(), &mut env),
        )
        .map_err(OnnxError::ApiInitFailed)?;
        eprintln!("[AI] Environment created");

        // -- Create session options --
        eprintln!("[AI] Creating session options...");
        let mut session_options: *mut OrtSessionOptions = std::ptr::null_mut();
        status_to_result(&api, (api.create_session_options())(&mut session_options))
            .map_err(OnnxError::SessionCreateFailed)?;

        // Use all available cores and enable graph optimizations
        let num_threads = num_cpus().max(1) as i32;
        eprintln!("[AI] Setting intra_op_num_threads={}", num_threads);
        let _ = status_to_result(
            &api,
            (api.set_intra_op_num_threads())(session_options, num_threads),
        );
        // ORT_ENABLE_ALL = 99
        let _ = status_to_result(
            &api,
            (api.set_session_graph_optimization_level())(session_options, 99),
        );
        eprintln!("[AI] Session options configured");

        // -- Create session (load model) --
        // On Windows, CreateSession expects a UTF-16 path
        eprintln!("[AI] Loading model (this may take a moment)...");
        let model_wide: Vec<u16> = model_path
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let mut session: *mut OrtSession = std::ptr::null_mut();
        let create_status =
            (api.create_session())(env, model_wide.as_ptr(), session_options, &mut session);
        if let Err(e) = status_to_result(&api, create_status) {
            (api.release_session_options())(session_options);
            (api.release_env())(env);
            return Err(OnnxError::ModelLoadFailed(e));
        }
        eprintln!("[AI] Model loaded successfully");

        // -- Get allocator --
        let mut allocator: *mut OrtAllocator = std::ptr::null_mut();
        status_to_result(
            &api,
            (api.get_allocator_with_default_options())(&mut allocator),
        )
        .map_err(|e| OnnxError::SessionCreateFailed(format!("Get allocator: {}", e)))?;

        // -- Auto-detect model input dimensions --
        eprintln!("[AI] Querying model input shape...");
        let (model_input_h, model_input_w) = {
            let mut type_info: *mut OrtTypeInfo = std::ptr::null_mut();
            let detected = if status_to_result(
                &api,
                (api.session_get_input_type_info())(session as *const _, 0, &mut type_info),
            )
            .is_ok()
                && !type_info.is_null()
            {
                let mut tensor_info: *const OrtTensorTypeAndShapeInfo = std::ptr::null();
                let size = if status_to_result(
                    &api,
                    (api.cast_type_info_to_tensor_info())(type_info as *const _, &mut tensor_info),
                )
                .is_ok()
                    && !tensor_info.is_null()
                {
                    let mut dim_count: usize = 0;
                    if status_to_result(
                        &api,
                        (api.get_dimensions_count())(tensor_info, &mut dim_count),
                    )
                    .is_ok()
                        && dim_count >= 3
                    {
                        let mut dims = vec![0i64; dim_count];
                        if status_to_result(
                            &api,
                            (api.get_dimensions())(tensor_info, dims.as_mut_ptr(), dim_count),
                        )
                        .is_ok()
                        {
                            eprintln!("[AI]   Model input shape: {:?}", dims);
                            // Input is typically [1, 3, H, W] (4D) or [3, H, W] (3D)
                            let (h, w) = if dim_count == 4 {
                                (dims[2], dims[3])
                            } else {
                                (dims[1], dims[2])
                            };
                            if h > 0 && w > 0 {
                                Some((h as u32, w as u32))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                    // NOTE: tensor_info is owned by type_info — do NOT release separately
                } else {
                    None
                };
                (api.release_type_info())(type_info);
                size
            } else {
                None
            };

            match detected {
                Some((h, w)) => {
                    eprintln!("[AI]   Auto-detected model input size: {}x{}", w, h);
                    (h, w)
                }
                None => {
                    eprintln!(
                        "[AI]   Could not detect model input size, defaulting to {}x{}",
                        DEFAULT_MODEL_SIZE, DEFAULT_MODEL_SIZE
                    );
                    (DEFAULT_MODEL_SIZE, DEFAULT_MODEL_SIZE)
                }
            }
        };
        // Use the larger dimension as the square input size (models use square inputs)
        let model_input_size = model_input_h.max(model_input_w);

        // -- Get input/output names and detect model profile --
        eprintln!("[AI] Querying input/output names...");
        let input_name = get_session_input_name(&api, session, 0, allocator)?;

        // Query the number of outputs
        let mut output_count: usize = 0;
        status_to_result(
            &api,
            (api.session_get_output_count())(session as *const _, &mut output_count),
        )
        .map_err(|e| OnnxError::SessionCreateFailed(format!("Get output count: {}", e)))?;
        eprintln!("[AI] Model has {} output(s)", output_count);

        // Detect model profile from input dimensions + output count
        let model_profile = ModelProfile::detect(model_input_h, model_input_w, output_count);
        eprintln!(
            "[AI] Detected model profile: {}",
            model_profile.description()
        );

        // Collect all output names
        let mut all_output_names = Vec::new();
        let mut all_output_name_cstrings = Vec::new();
        for i in 0..output_count {
            let name = get_session_output_name(&api, session, i, allocator)?;
            eprintln!("[AI]   Output {}: '{}'", i, name);
            all_output_name_cstrings.push(std::ffi::CString::new(name.clone()).unwrap());
            all_output_names.push(name);
        }
        let output_name_ptrs: Vec<*const std::ffi::c_char> = all_output_name_cstrings
            .iter()
            .map(|cs| cs.as_ptr())
            .collect();

        eprintln!(
            "[AI] Input: '{}' ({}x{}), {} output(s)",
            input_name, model_input_size, model_input_size, output_count
        );

        // -- Preprocess --
        eprintln!(
            "[AI] Preprocessing image (resize to {}x{}, normalize)...",
            model_input_size, model_input_size
        );
        let (orig_w, orig_h) = input.dimensions();
        let mut tensor_data = preprocess_image(input, model_input_size);
        let tensor_shape: [i64; 4] = [1, 3, model_input_size as i64, model_input_size as i64];

        // -- Create memory info --
        eprintln!("[AI] Creating memory info and input tensor...");
        let mut memory_info: *mut OrtMemoryInfo = std::ptr::null_mut();
        status_to_result(
            &api,
            (api.create_cpu_memory_info())(
                OrtAllocatorType::ArenaAllocator,
                OrtMemType::Default,
                &mut memory_info,
            ),
        )
        .map_err(|e| OnnxError::InferenceFailed(format!("Create memory info: {}", e)))?;

        // -- Create input tensor --
        let mut input_tensor: *mut OrtValue = std::ptr::null_mut();
        let data_len = tensor_data.len() * std::mem::size_of::<f32>();
        status_to_result(
            &api,
            (api.create_tensor_with_data())(
                memory_info,
                tensor_data.as_mut_ptr() as *mut std::ffi::c_void,
                data_len,
                tensor_shape.as_ptr(),
                4,
                ONNXTensorElementDataType::Float,
                &mut input_tensor,
            ),
        )
        .map_err(|e| OnnxError::InferenceFailed(format!("Create input tensor: {}", e)))?;

        // -- Run inference requesting ALL outputs --
        eprintln!(
            "[AI] Running inference (requesting {} output(s))...",
            output_count
        );
        let input_name_c = std::ffi::CString::new(input_name.clone()).unwrap();
        let input_names = [input_name_c.as_ptr()];
        let input_tensors = [input_tensor as *const OrtValue];
        let mut output_tensors: Vec<*mut OrtValue> = vec![std::ptr::null_mut(); output_count];

        let run_status = (api.run())(
            session,
            std::ptr::null(), // run_options
            input_names.as_ptr(),
            input_tensors.as_ptr(),
            1,
            output_name_ptrs.as_ptr(),
            output_count,
            output_tensors.as_mut_ptr(),
        );

        if let Err(e) = status_to_result(&api, run_status) {
            eprintln!("[AI] Inference FAILED: {}", e);
            // Cleanup before returning error
            (api.release_value())(input_tensor);
            (api.release_memory_info())(memory_info);
            (api.release_session())(session);
            (api.release_session_options())(session_options);
            (api.release_env())(env);
            return Err(OnnxError::InferenceFailed(e));
        }

        // -- Auto-select best output by confidence scoring --
        eprintln!(
            "[AI] Inference completed. Scoring {} output(s) to find best mask...",
            output_count
        );

        // Score all outputs: for each output, compute mask dimensions and confidence.
        // The most refined decoder stage has the most decisive (bimodal) probability distribution.
        struct OutputInfo {
            index: usize,
            dims: Vec<i64>,
            out_h: u32,
            out_w: u32,
            confidence: f64,
        }

        let mut output_infos: Vec<OutputInfo> = Vec::new();
        for (i, &ot) in output_tensors.iter().enumerate() {
            if ot.is_null() {
                continue;
            }

            let mut ti: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            if status_to_result(
                &api,
                (api.get_tensor_type_and_shape())(ot as *const _, &mut ti),
            )
            .is_err()
            {
                continue;
            }

            let mut dc: usize = 0;
            let _ = status_to_result(&api, (api.get_dimensions_count())(ti, &mut dc));
            let mut ds = vec![0i64; dc];
            let _ = status_to_result(&api, (api.get_dimensions())(ti, ds.as_mut_ptr(), dc));
            (api.release_tensor_type_and_shape_info())(ti);

            // Parse spatial dims
            let (oh, ow) = match ds.len() {
                4 => (ds[2] as u32, ds[3] as u32),
                3 => (ds[1] as u32, ds[2] as u32),
                _ => {
                    let total: i64 = ds.iter().product();
                    let side = (total as f64).sqrt() as u32;
                    (side, side)
                }
            };

            // Get data pointer and compute confidence score
            let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            if status_to_result(&api, (api.get_tensor_mutable_data())(ot, &mut data_ptr)).is_ok()
                && !data_ptr.is_null()
            {
                let total = (oh as usize) * (ow as usize);
                if total > 0 {
                    let slice = std::slice::from_raw_parts(data_ptr as *const f32, total);
                    let score = mask_confidence_score(slice);
                    eprintln!(
                        "[AI]   Output {} '{}': dims {:?} ({}x{}), confidence {:.4}",
                        i,
                        all_output_names.get(i).unwrap_or(&"?".to_string()),
                        ds,
                        ow,
                        oh,
                        score
                    );
                    output_infos.push(OutputInfo {
                        index: i,
                        dims: ds,
                        out_h: oh,
                        out_w: ow,
                        confidence: score,
                    });
                }
            }
        }

        // Select the best output:
        // 1. Find the one with the highest confidence score
        // 2. If scores are very close (within 1%), prefer the model profile's preferred index
        let preferred_idx = model_profile.preferred_output_index(output_count);
        let best_info = if output_infos.is_empty() {
            return Err(OnnxError::InvalidOutput(
                "No valid outputs found".to_string(),
            ));
        } else {
            let max_confidence = output_infos
                .iter()
                .map(|o| o.confidence)
                .fold(0.0f64, f64::max);
            // Among outputs with confidence within 1% of the best, prefer the model's default
            let close_to_best: Vec<&OutputInfo> = output_infos
                .iter()
                .filter(|o| o.confidence >= max_confidence - 0.01)
                .collect();

            if let Some(preferred) = close_to_best.iter().find(|o| o.index == preferred_idx) {
                preferred
            } else {
                close_to_best
                    .into_iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                    .unwrap()
            }
        };

        let best_output_idx = best_info.index;
        let (out_h, out_w) = (best_info.out_h, best_info.out_w);
        eprintln!(
            "[AI] Selected output {} (confidence {:.4}) — {}x{}",
            best_output_idx, best_info.confidence, out_w, out_h
        );

        // Get pointer to best output's data
        let output_tensor = output_tensors[best_output_idx];
        let mut out_data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        status_to_result(
            &api,
            (api.get_tensor_mutable_data())(output_tensor, &mut out_data_ptr),
        )
        .map_err(|e| OnnxError::InvalidOutput(format!("Get tensor data: {}", e)))?;

        let total_elements = (out_h * out_w) as usize;
        let out_slice = std::slice::from_raw_parts(out_data_ptr as *const f32, total_elements);

        // -- Detect output type and convert to probabilities --
        let is_prob = is_probability_space(out_slice);
        eprintln!(
            "[AI] Output value space: {}",
            if is_prob {
                "probabilities [0,1]"
            } else {
                "logits (applying sigmoid)"
            }
        );

        let probabilities: Vec<f32> = out_slice
            .iter()
            .map(|&v| to_probability(v, is_prob))
            .collect();

        // Log mask statistics for debugging
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        let mut sum_prob = 0.0f64;
        for &p in probabilities.iter() {
            min_val = min_val.min(p);
            max_val = max_val.max(p);
            sum_prob += p as f64;
        }
        let avg_prob = sum_prob / total_elements as f64;
        eprintln!(
            "[AI] Mask stats: probability range [{:.4}, {:.4}], avg {:.4}",
            min_val, max_val, avg_prob
        );

        // -- Postprocess --
        eprintln!(
            "[AI] Postprocessing mask (applying to {}x{} image)...",
            orig_w, orig_h
        );
        eprintln!(
            "[AI]   Settings: threshold={:.2}, edge_feather={:.1}, mask_expansion={}, smooth_edges={}, fill_holes={}",
            settings.threshold,
            settings.edge_feather,
            settings.mask_expansion,
            settings.smooth_edges,
            settings.fill_holes
        );
        let result = postprocess_mask(&probabilities, out_h, out_w, input, settings);

        // -- Cleanup --
        eprintln!("[AI] Cleaning up ONNX resources...");
        for &ot in &output_tensors {
            if !ot.is_null() {
                (api.release_value())(ot);
            }
        }
        (api.release_value())(input_tensor);
        (api.release_memory_info())(memory_info);
        (api.release_session())(session);
        (api.release_session_options())(session_options);
        (api.release_env())(env);

        // If the original was smaller and we want to preserve size
        if result.dimensions() != (orig_w, orig_h) {
            eprintln!("[AI] Resizing result to match original dimensions");
            Ok(image::imageops::resize(
                &result,
                orig_w,
                orig_h,
                image::imageops::FilterType::Lanczos3,
            ))
        } else {
            eprintln!(
                "[AI] Done! Result is {}x{}",
                result.width(),
                result.height()
            );
            Ok(result)
        }
    }
}

/// Get the name of an input tensor from the session.
unsafe fn get_session_input_name(
    api: &OrtApi,
    session: *mut OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> Result<String, OnnxError> {
    let mut name_ptr: *mut std::ffi::c_char = std::ptr::null_mut();
    status_to_result(
        api,
        (api.session_get_input_name())(session as *const _, index, allocator, &mut name_ptr),
    )
    .map_err(|e| OnnxError::SessionCreateFailed(format!("Get input name: {}", e)))?;

    let name = if name_ptr.is_null() {
        "input".to_string()
    } else {
        let s = std::ffi::CStr::from_ptr(name_ptr)
            .to_string_lossy()
            .into_owned();
        (api.allocator_free())(allocator, name_ptr as *mut std::ffi::c_void);
        s
    };
    Ok(name)
}

/// Get the name of an output tensor from the session.
unsafe fn get_session_output_name(
    api: &OrtApi,
    session: *mut OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> Result<String, OnnxError> {
    let mut name_ptr: *mut std::ffi::c_char = std::ptr::null_mut();
    status_to_result(
        api,
        (api.session_get_output_name())(session as *const _, index, allocator, &mut name_ptr),
    )
    .map_err(|e| OnnxError::SessionCreateFailed(format!("Get output name: {}", e)))?;

    let name = if name_ptr.is_null() {
        "output".to_string()
    } else {
        let s = std::ffi::CStr::from_ptr(name_ptr)
            .to_string_lossy()
            .into_owned();
        (api.allocator_free())(allocator, name_ptr as *mut std::ffi::c_void);
        s
    };
    Ok(name)
}

/// Get the number of logical CPU cores (simple heuristic).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
