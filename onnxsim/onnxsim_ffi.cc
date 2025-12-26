/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnxsim_ffi.h"

#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "onnxsim.h"

namespace {
// Thread-local storage for last error message
thread_local std::string g_last_error;

void set_last_error(const std::string& error) {
  g_last_error = error;
}

onnxsim_error_t handle_exception() {
  try {
    throw;  // Re-throw the caught exception
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return ONNXSIM_ERROR_INTERNAL;
  } catch (...) {
    set_last_error("Unknown error");
    return ONNXSIM_ERROR_INTERNAL;
  }
}
}  // namespace

void onnxsim_init_env(void) {
  InitEnv();
}

onnxsim_error_t onnxsim_simplify_bytes(
    const uint8_t* model_bytes,
    size_t model_bytes_len,
    const char** skip_optimizers,
    size_t skip_optimizers_len,
    int constant_folding,
    int shape_inference,
    size_t tensor_size_threshold,
    uint8_t** out_bytes,
    size_t* out_bytes_len) {
  try {
    // Validate arguments
    if (model_bytes == nullptr) {
      set_last_error("model_bytes cannot be NULL");
      return ONNXSIM_ERROR_INVALID_ARGUMENT;
    }
    if (out_bytes == nullptr || out_bytes_len == nullptr) {
      set_last_error("out_bytes or out_bytes_len cannot be NULL");
      return ONNXSIM_ERROR_INVALID_ARGUMENT;
    }

    // Parse model from bytes
    onnx::ModelProto model;
    const std::string model_str(reinterpret_cast<const char*>(model_bytes),
                                model_bytes_len);
    if (!model.ParseFromString(model_str)) {
      set_last_error("Failed to parse model protobuf");
      return ONNXSIM_ERROR_PARSE_FAILED;
    }

    // Prepare skip optimizers
    std::optional<std::vector<std::string>> skip_opts;
    if (skip_optimizers != nullptr && skip_optimizers_len > 0) {
      std::vector<std::string> opts;
      for (size_t i = 0; i < skip_optimizers_len; ++i) {
        if (skip_optimizers[i] != nullptr) {
          opts.push_back(std::string(skip_optimizers[i]));
        }
      }
      skip_opts = std::move(opts);
    }

    // Simplify model
    auto simplified_model = Simplify(
        model, skip_opts, constant_folding != 0, shape_inference != 0,
        tensor_size_threshold);

    // Serialize output
    std::string output;
    if (!simplified_model.SerializeToString(&output)) {
      set_last_error("Failed to serialize simplified model");
      return ONNXSIM_ERROR_SERIALIZE_FAILED;
    }

    // Allocate and copy output
    *out_bytes_len = output.size();
    *out_bytes = static_cast<uint8_t*>(std::malloc(*out_bytes_len));
    if (*out_bytes == nullptr) {
      set_last_error("Failed to allocate memory for output");
      return ONNXSIM_ERROR_INTERNAL;
    }
    std::memcpy(*out_bytes, output.data(), *out_bytes_len);

    return ONNXSIM_SUCCESS;
  } catch (...) {
    return handle_exception();
  }
}

onnxsim_error_t onnxsim_simplify_file(
    const char* in_path,
    const char* out_path,
    const char** skip_optimizers,
    size_t skip_optimizers_len,
    int constant_folding,
    int shape_inference,
    size_t tensor_size_threshold) {
  try {
    // Validate arguments
    if (in_path == nullptr || out_path == nullptr) {
      set_last_error("in_path and out_path cannot be NULL");
      return ONNXSIM_ERROR_INVALID_ARGUMENT;
    }

    // Prepare skip optimizers
    std::optional<std::vector<std::string>> skip_opts;
    if (skip_optimizers != nullptr && skip_optimizers_len > 0) {
      std::vector<std::string> opts;
      for (size_t i = 0; i < skip_optimizers_len; ++i) {
        if (skip_optimizers[i] != nullptr) {
          opts.push_back(std::string(skip_optimizers[i]));
        }
      }
      skip_opts = std::move(opts);
    }

    // Simplify file
    SimplifyPath(std::string(in_path), std::string(out_path), skip_opts,
                 constant_folding != 0, shape_inference != 0,
                 tensor_size_threshold);

    return ONNXSIM_SUCCESS;
  } catch (...) {
    return handle_exception();
  }
}

void onnxsim_free_string(void* ptr) {
  std::free(ptr);
}

const char* onnxsim_get_last_error(void) {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}
