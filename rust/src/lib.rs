#![allow(non_camel_case_types, non_upper_case_globals)]

// Include the auto-generated FFI bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::{CStr, CString};
use std::ptr;
use thiserror::Error;

/// Error type for ONNX simplifier operations
#[derive(Error, Debug)]
pub enum OnnxSimError {
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Failed to parse model protobuf: {0}")]
    ParseFailed(String),

    #[error("Failed to serialize model protobuf: {0}")]
    SerializeFailed(String),

    #[error("Simplification failed: {0}")]
    SimplificationFailed(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Nul error: {0}")]
    NulError(#[from] std::ffi::NulError),
}

impl From<OnnxSimError> for String {
    fn from(err: OnnxSimError) -> Self {
        err.to_string()
    }
}

/// Result type for ONNX simplifier operations
pub type Result<T> = std::result::Result<T, OnnxSimError>;

/// Configuration options for model simplification
#[derive(Debug, Clone, Default)]
pub struct SimplifyOptions {
    /// List of optimizers to skip
    pub skip_optimizers: Option<Vec<String>>,

    /// Enable constant folding
    pub constant_folding: bool,

    /// Enable shape inference
    pub shape_inference: bool,

    /// Tensor size threshold for optimization
    pub tensor_size_threshold: usize,
}

impl SimplifyOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_constant_folding(mut self, enabled: bool) -> Self {
        self.constant_folding = enabled;
        self
    }

    pub fn with_shape_inference(mut self, enabled: bool) -> Self {
        self.shape_inference = enabled;
        self
    }

    pub fn with_skip_optimizers(mut self, optimizers: Vec<String>) -> Self {
        self.skip_optimizers = Some(optimizers);
        self
    }

    pub fn with_tensor_size_threshold(mut self, threshold: usize) -> Self {
        self.tensor_size_threshold = threshold;
        self
    }
}

/// Initialize the ONNX environment
///
/// Must be called before any other onnxsim functions.
pub fn init_env() {
    unsafe {
        onnxsim_init_env();
    }
}

/// Simplify an ONNX model from bytes
///
/// # Arguments
///
/// * `model_bytes` - The serialized model protobuf bytes
/// * `options` - Simplification options
///
/// # Returns
///
/// The simplified model as bytes
///
/// # Example
///
/// ```no_run
/// use onnxsim::{init_env, simplify_bytes, SimplifyOptions};
///
/// init_env();
/// let model_bytes = std::fs::read("model.onnx").unwrap();
/// let simplified = simplify_bytes(&model_bytes, SimplifyOptions::default()).unwrap();
/// std::fs::write("simplified.onnx", simplified).unwrap();
/// ```
pub fn simplify_bytes(model_bytes: &[u8], options: SimplifyOptions) -> Result<Vec<u8>> {
    init_env();

    // Prepare skip optimizers
    let skip_optimizers = options.skip_optimizers.unwrap_or_default();
    let skip_optimizers_cstrings: Result<Vec<CString>> = skip_optimizers
        .iter()
        .map(|s| CString::new(s.as_str()).map_err(|e| OnnxSimError::InvalidArgument(e.to_string())))
        .collect();
    let skip_optimizers_cstrings = skip_optimizers_cstrings?;

    let skip_optimizers_ptrs: Vec<*const i8> = skip_optimizers_cstrings
        .iter()
        .map(|s| s.as_ptr())
        .collect();

    let skip_optimizers_ptr: *mut *const i8 = if skip_optimizers_ptrs.is_empty() {
        ptr::null_mut()
    } else {
        skip_optimizers_ptrs.as_ptr() as *mut *const i8
    };

    let mut out_bytes: *mut u8 = ptr::null_mut();
    let mut out_bytes_len: usize = 0;

    let result = unsafe {
        onnxsim_simplify_bytes(
            model_bytes.as_ptr(),
            model_bytes.len(),
            skip_optimizers_ptr,
            skip_optimizers_ptrs.len(),
            options.constant_folding as i32,
            options.shape_inference as i32,
            options.tensor_size_threshold,
            &mut out_bytes,
            &mut out_bytes_len,
        )
    };

    if result != onnxsim_error_t_ONNXSIM_SUCCESS {
        let error_msg = unsafe {
            let error_ptr = onnxsim_get_last_error();
            if error_ptr.is_null() {
                String::from("Unknown error")
            } else {
                CStr::from_ptr(error_ptr).to_string_lossy().into_owned()
            }
        };

        return Err(match result {
            onnxsim_error_t_ONNXSIM_ERROR_INVALID_ARGUMENT => {
                OnnxSimError::InvalidArgument(error_msg)
            }
            onnxsim_error_t_ONNXSIM_ERROR_PARSE_FAILED => OnnxSimError::ParseFailed(error_msg),
            onnxsim_error_t_ONNXSIM_ERROR_SERIALIZE_FAILED => {
                OnnxSimError::SerializeFailed(error_msg)
            }
            onnxsim_error_t_ONNXSIM_ERROR_SIMPLIFICATION_FAILED => {
                OnnxSimError::SimplificationFailed(error_msg)
            }
            _ => OnnxSimError::Internal(error_msg),
        });
    }

    // Copy the output bytes
    let output = unsafe {
        if out_bytes.is_null() || out_bytes_len == 0 {
            return Err(OnnxSimError::Internal("Empty output".to_string()));
        }
        std::slice::from_raw_parts(out_bytes, out_bytes_len).to_vec()
    };

    // Free the allocated memory
    unsafe {
        onnxsim_free_string(out_bytes as *mut _);
    }

    Ok(output)
}

/// Simplify an ONNX model from file path
///
/// # Arguments
///
/// * `in_path` - Path to the input ONNX model file
/// * `out_path` - Path to save the simplified ONNX model
/// * `options` - Simplification options
///
/// # Example
///
/// ```no_run
/// use onnxsim::{init_env, simplify_file, SimplifyOptions};
///
/// init_env();
/// simplify_file("model.onnx", "simplified.onnx", SimplifyOptions::default()).unwrap();
/// ```
pub fn simplify_file<P: AsRef<std::path::Path>>(
    in_path: P,
    out_path: P,
    options: SimplifyOptions,
) -> Result<()> {
    init_env();

    let in_path_str = in_path
        .as_ref()
        .to_str()
        .ok_or_else(|| OnnxSimError::InvalidArgument("Invalid UTF-8 in input path".to_string()))?;
    let in_path_cstring = CString::new(in_path_str)?;

    let out_path_str = out_path
        .as_ref()
        .to_str()
        .ok_or_else(|| OnnxSimError::InvalidArgument("Invalid UTF-8 in output path".to_string()))?;
    let out_path_cstring = CString::new(out_path_str)?;

    // Prepare skip optimizers
    let skip_optimizers = options.skip_optimizers.unwrap_or_default();
    let skip_optimizers_cstrings: Result<Vec<CString>> = skip_optimizers
        .iter()
        .map(|s| CString::new(s.as_str()).map_err(|e| OnnxSimError::InvalidArgument(e.to_string())))
        .collect();
    let skip_optimizers_cstrings = skip_optimizers_cstrings?;

    let skip_optimizers_ptrs: Vec<*const i8> = skip_optimizers_cstrings
        .iter()
        .map(|s| s.as_ptr())
        .collect();

    let skip_optimizers_ptr: *mut *const i8 = if skip_optimizers_ptrs.is_empty() {
        ptr::null_mut()
    } else {
        skip_optimizers_ptrs.as_ptr() as *mut *const i8
    };

    let result = unsafe {
        onnxsim_simplify_file(
            in_path_cstring.as_ptr(),
            out_path_cstring.as_ptr(),
            skip_optimizers_ptr,
            skip_optimizers_ptrs.len(),
            options.constant_folding as i32,
            options.shape_inference as i32,
            options.tensor_size_threshold,
        )
    };

    if result != onnxsim_error_t_ONNXSIM_SUCCESS {
        let error_msg = unsafe {
            let error_ptr = onnxsim_get_last_error();
            if error_ptr.is_null() {
                String::from("Unknown error")
            } else {
                CStr::from_ptr(error_ptr).to_string_lossy().into_owned()
            }
        };

        return Err(match result {
            onnxsim_error_t_ONNXSIM_ERROR_INVALID_ARGUMENT => {
                OnnxSimError::InvalidArgument(error_msg)
            }
            onnxsim_error_t_ONNXSIM_ERROR_PARSE_FAILED => OnnxSimError::ParseFailed(error_msg),
            onnxsim_error_t_ONNXSIM_ERROR_SERIALIZE_FAILED => {
                OnnxSimError::SerializeFailed(error_msg)
            }
            onnxsim_error_t_ONNXSIM_ERROR_SIMPLIFICATION_FAILED => {
                OnnxSimError::SimplificationFailed(error_msg)
            }
            _ => OnnxSimError::Internal(error_msg),
        });
    }

    Ok(())
}
