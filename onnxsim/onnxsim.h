#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <onnx/onnx_pb.h>

// Structure to record operator folding information
struct FoldedOp {
  std::string op_type;
  std::string op_name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  bool success;
  std::string error_msg;
};

struct FoldingRecord {
  std::vector<FoldedOp> folded_ops;
  size_t total_attempted;
  size_t total_succeeded;
  size_t total_failed;

  FoldingRecord() : total_attempted(0), total_succeeded(0), total_failed(0) {}

  void RecordFold(const FoldedOp& op) {
    folded_ops.push_back(op);
    total_attempted++;
    if (op.success) {
      total_succeeded++;
    } else {
      total_failed++;
    }
  }

  void Clear() {
    folded_ops.clear();
    total_attempted = 0;
    total_succeeded = 0;
    total_failed = 0;
  }
};

// Global folding record
extern FoldingRecord g_folding_record;

struct ModelExecutor {
  virtual ~ModelExecutor() = default;
  static void set_instance(std::shared_ptr<const ModelExecutor> instance) {
    instance_ = std::move(instance);
  }
  static std::vector<onnx::TensorProto> Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) {
    if (instance_ == nullptr) {
      throw std::runtime_error("empty instance");
    }
    return instance_->_Run(model, inputs);
  }

  // public it for pybind11
  virtual std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const = 0;

 private:
  static std::shared_ptr<const ModelExecutor> instance_;
};

void InitEnv();

onnx::ModelProto Simplify(
    const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold);

void SimplifyPath(const std::string& in_path, const std::string& out_path,
                  std::optional<std::vector<std::string>> skip_optimizers,
                  bool constant_folding, bool shape_inference,
                  size_t tensor_size_threshold);

// Get the global folding record
const FoldingRecord& GetFoldingRecord();

// Clear the global folding record
void ClearFoldingRecord();
