/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Error codes
typedef enum {
  ONNXSIM_SUCCESS = 0,
  ONNXSIM_ERROR_INVALID_ARGUMENT = 1,
  ONNXSIM_ERROR_PARSE_FAILED = 2,
  ONNXSIM_ERROR_SERIALIZE_FAILED = 3,
  ONNXSIM_ERROR_SIMPLIFICATION_FAILED = 4,
  ONNXSIM_ERROR_INTERNAL = 5
} onnxsim_error_t;

// Handle type for opaque objects
typedef void* onnxsim_handle_t;

/**
 * Initialize the ONNX environment.
 * Must be called before any other onnxsim functions.
 */
void onnxsim_init_env(void);

/**
 * Simplify an ONNX model from bytes.
 *
 * @param model_bytes Pointer to the serialized model protobuf bytes
 * @param model_bytes_len Length of the model bytes
 * @param skip_optimizers Array of optimizer names to skip (NULL for none)
 * @param skip_optimizers_len Length of skip_optimizers array
 * @param constant_folding Enable constant folding (1=enabled, 0=disabled)
 * @param shape_inference Enable shape inference (1=enabled, 0=disabled)
 * @param tensor_size_threshold Tensor size threshold for optimization
 * @param out_bytes Pointer to receive output bytes (must be freed with onnxsim_free_string)
 * @param out_bytes_len Pointer to receive output bytes length
 * @return Error code (ONNXSIM_SUCCESS on success)
 */
onnxsim_error_t onnxsim_simplify_bytes(
    const uint8_t* model_bytes,
    size_t model_bytes_len,
    const char** skip_optimizers,
    size_t skip_optimizers_len,
    int constant_folding,
    int shape_inference,
    size_t tensor_size_threshold,
    uint8_t** out_bytes,
    size_t* out_bytes_len);

/**
 * Simplify an ONNX model from file path.
 *
 * @param in_path Path to the input ONNX model file
 * @param out_path Path to save the simplified ONNX model
 * @param skip_optimizers Array of optimizer names to skip (NULL for none)
 * @param skip_optimizers_len Length of skip_optimizers array
 * @param constant_folding Enable constant folding (1=enabled, 0=disabled)
 * @param shape_inference Enable shape inference (1=enabled, 0=disabled)
 * @param tensor_size_threshold Tensor size threshold for optimization
 * @return Error code (ONNXSIM_SUCCESS on success)
 */
onnxsim_error_t onnxsim_simplify_file(
    const char* in_path,
    const char* out_path,
    const char** skip_optimizers,
    size_t skip_optimizers_len,
    int constant_folding,
    int shape_inference,
    size_t tensor_size_threshold);

/**
 * Free a string/bytes allocated by onnxsim functions.
 *
 * @param ptr Pointer to the string/bytes to free
 */
void onnxsim_free_string(void* ptr);

/**
 * Get the last error message.
 * The returned string is valid until the next call to onnxsim functions.
 *
 * @return Error message string (NULL if no error)
 */
const char* onnxsim_get_last_error(void);

#ifdef __cplusplus
}
#endif
