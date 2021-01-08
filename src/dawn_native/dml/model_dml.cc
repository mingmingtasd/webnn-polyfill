#include "dawn_native/dml/model_dml.h"

#include "common/Assert.h"
#include "common/Log.h"
#include "dawn_native/ErrorData.h"
#include "dawn_native/dml/compilation_dml.h"

namespace dawn_native {
namespace dml {

namespace {
bool GetDmlTensorDataType(
    wnn::OperandType operand_type, DML_TENSOR_DATA_TYPE& dml_tensor_data_type) {
  if (operand_type == wnn::OperandType::Float32) {
    dml_tensor_data_type = DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if (operand_type == wnn::OperandType::Float16) {
    dml_tensor_data_type = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if (operand_type == wnn::OperandType::Int32) {
    dml_tensor_data_type = DML_TENSOR_DATA_TYPE_INT32;
  } else if (operand_type == wnn::OperandType::Uint32) {
    dml_tensor_data_type = DML_TENSOR_DATA_TYPE_UINT32;
  } else {
    return false;
  }
  return true;
}

bool GetDmlTensorDimensions(
    int32_t const * dimensions, uint32_t dimensions_count,
    ::dml::TensorDimensions& dml_tensor_dimensions) {
  if (dimensions_count > DML_TENSOR_DIMENSION_COUNT_MAX) {
    dawn::ErrorLog() << "Tensor dimension count " << dimensions_count
                     << " is greater than DML_TENSOR_DIMENSION_COUNT_MAX "
                     << DML_TENSOR_DIMENSION_COUNT_MAX;
    return false;
  }
  dml_tensor_dimensions.resize(dimensions_count);
  for (uint32_t i = 0; i < dimensions_count; ++i) {
    int32_t d = dimensions[i];
    if (d < 0) {
      dawn::ErrorLog() << "DML doesn't support the negative dimension value";
      return false;
    }
    dml_tensor_dimensions[i] = d;
  }
  return true;
}

::dml::TensorDimensions ExpandDimensions(
    const ::dml::TensorDimensions& dims, size_t rank) {
  DAWN_ASSERT(rank >= dims.size());
  ::dml::TensorDimensions new_dims(rank, 1);
  for (size_t i = 0; i < dims.size(); ++i) {
    new_dims[new_dims.size() - i - 1] = dims[dims.size() - i - 1];
  }
  return new_dims;
}

::dml::TensorDimensions ShrinkDimensions(
    const ::dml::TensorDimensions& dims, size_t rank) {
  DAWN_ASSERT(rank <= dims.size());
  ::dml::TensorDimensions new_dims(rank);
  for (size_t i = 0; i < rank; ++i) {
    new_dims[new_dims.size() - i - 1] = dims[dims.size() - i - 1];
  }
  return new_dims;
}

// Refer to
// https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
::dml::TensorDimensions CalculateStrides(
    const ::dml::TensorDimensions& dims,
    const std::vector<bool>& broadcast = {}) {
  size_t rank = dims.size();
  ::dml::TensorDimensions strides(rank, 1);
  for (size_t i = 1; i < rank; i++) {
    size_t j = dims.size() - i - 1;
    strides[j] = dims[j + 1] * strides[j + 1];
  }
  if (!broadcast.empty()) {
    for (size_t i = 0; i < rank; i++) {
      if (broadcast[i]) {
        strides[i] = 0;
      }
    }
  }
  return strides;
}

bool BroadcastDimensions(
    const ::dml::TensorDimensions& a_dims,
    const ::dml::TensorDimensions& b_dims,
    bool& a_broadcasted,
    ::dml::TensorDimensions& a_new_dims,
    ::dml::TensorDimensions& a_new_strides,
    bool& b_broadcasted,
    ::dml::TensorDimensions& b_new_dims,
    ::dml::TensorDimensions& b_new_strides,
    size_t skip_axis = 0) {
  auto a_rank = a_dims.size();
  auto b_rank = b_dims.size();
  auto new_rank = std::max(a_rank, b_rank);
  a_new_dims.resize(new_rank);
  a_new_strides.resize(new_rank);
  std::vector<bool> a_broadcast(new_rank, false);
  b_new_dims.resize(new_rank);
  b_new_strides.resize(new_rank);
  std::vector<bool> b_broadcast(new_rank, false);
  if (new_rank > a_rank) {
    a_new_dims = ExpandDimensions(a_dims, new_rank);
    a_broadcasted = true;
  } else {
    a_new_dims = a_dims;
  }
  if (new_rank > b_rank) {
    b_new_dims = ExpandDimensions(b_dims, new_rank);
    b_broadcasted = true;
  } else {
    b_new_dims = b_dims;
  }
  for (size_t i = 0; i < new_rank - skip_axis; i++) {
    if (a_new_dims[i] == 1 && b_new_dims[i] != 1) {
      a_new_dims[i] = b_new_dims[i];
      a_broadcast[i] = true;
      a_broadcasted = true;
    } else if (b_new_dims[i] == 1 && a_new_dims[i] != 1) {
      b_new_dims[i] = a_new_dims[i];
      b_broadcast[i] = true;
      b_broadcasted = true;
    } else if (a_new_dims[i] != b_new_dims[i]) {
      return false;
    }
  }
  a_new_strides = CalculateStrides(a_new_dims, a_broadcast);
  b_new_strides = CalculateStrides(b_new_dims, b_broadcast);
  return true;
}

std::string OpTypeToString(op::BinaryOpType type) {
  if (type == op::BinaryOpType::kAdd) {
    return "add";
  } else if (type == op::BinaryOpType::kMul) {
    return "mul";
  } else if (type == op::BinaryOpType::kSub) {
    return "sub";
  } else if (type == op::BinaryOpType::kDiv) {
    return "div";
  } else if (type == op::BinaryOpType::kMatMul) {
    return "matmul";
  }
  return std::to_string(type);
}

std::string OpTypeToString(op::UnaryOpType type) {
  if (type == op::UnaryOpType::kRelu) {
    return "relu";
  } else if (type == op::UnaryOpType::kSoftmax) {
    return "softmax";
  }
  return std::to_string(type);
}

}  // namespace

std::string DmlTensorDimensionsToString(
    const ::dml::TensorDimensions& dimensions) {
  std::string output = "[";
  for(size_t i = 0; i < dimensions.size(); ++i) {
    output.append(std::to_string(dimensions[i]));
    if (i != dimensions.size() - 1) {
      output.append(",");
    }
  }
  output.append("]");
  return output;
}

std::string DmlTensorDataTypeToString(DML_TENSOR_DATA_TYPE type) {
  if (type == DML_TENSOR_DATA_TYPE_UNKNOWN) {
    return "UNKNOWN";
  } else if (type == DML_TENSOR_DATA_TYPE_FLOAT32) {
    return "FLOAT32";
  } else if (type == DML_TENSOR_DATA_TYPE_FLOAT16) {
    return "FLOAT16";
  } else if (type == DML_TENSOR_DATA_TYPE_UINT32) {
    return "UINT32";
  } else if (type == DML_TENSOR_DATA_TYPE_UINT16) {
    return "UINT16";
  } else if (type == DML_TENSOR_DATA_TYPE_UINT8) {
    return "UINT8";
  } else if (type == DML_TENSOR_DATA_TYPE_INT32) {
    return "INT32";
  } else if (type == DML_TENSOR_DATA_TYPE_INT16) {
    return "INT16";
  } else if (type == DML_TENSOR_DATA_TYPE_INT8) {
    return "INT8";
  } else if (type == DML_TENSOR_DATA_TYPE_FLOAT64) {
    return "FLOAT64";
  } else if (type == DML_TENSOR_DATA_TYPE_UINT64) {
    return "UINT64";
  } else if (type == DML_TENSOR_DATA_TYPE_INT64) {
    return "INT64";
  }
  return std::to_string(type);
}

Model::Model(ModelBuilder *model_builder) : ModelBase(model_builder) {
#if defined(_DEBUG)
  device_.reset(new ::pydml::Device(true, true));
#else
  device_.reset(new ::pydml::Device(true, false));
#endif
  graph_.reset(new ::dml::Graph(device_->GetDevice()));
}

MaybeError Model::AddConstant(const op::Constant *constant) {
  const OperandDescriptor* desc = constant->GetOperandDescriptor();
  DML_TENSOR_DATA_TYPE dml_tensor_type;
  if (!GetDmlTensorDataType(desc->type, dml_tensor_type)) {
    return DAWN_INTERNAL_ERROR("Failed to get DML tensor type.");
  }
  ::dml::TensorDimensions dml_tensor_dims;
  if (!GetDmlTensorDimensions(
      desc->dimensions, desc->dimensionsCount, dml_tensor_dims)) {
    return DAWN_INTERNAL_ERROR("Failed to get DML tensor dimensions.");
  }
  ::dml::TensorDesc dml_tensor_desc(
      dml_tensor_type, ::DML_TENSOR_FLAGS::DML_TENSOR_FLAG_OWNED_BY_DML,
      dml_tensor_dims, ::dml::TensorPolicy::Default());
  ::dml::Expression dml_constant =
      ::dml::InputTensor(*graph_, bindings_.size(), dml_tensor_desc);
  expressions_.insert(std::make_pair(constant, dml_constant));
  std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(
      dml_constant, const_cast<void*>(constant->GetValue()),
      constant->GetSize()));
  bindings_.push_back(std::move(binding));
  DAWN_DEBUG() << " impl: " << dml_constant.Impl()
               << " value: " << constant->GetValue()
               << " size: " << constant->GetSize()
               << ", type: "
               << DmlTensorDataTypeToString(
                  dml_constant.GetOutputDesc().dataType)
               << ", dimensions: "
               << DmlTensorDimensionsToString(
                  dml_constant.GetOutputDesc().sizes);
  return {};
}

MaybeError Model::AddInput(const op::Input *input) {
  const OperandDescriptor* desc = input->GetOperandDescriptor();
  DML_TENSOR_DATA_TYPE dml_tensor_type;
  if (!GetDmlTensorDataType(desc->type, dml_tensor_type)) {
    return DAWN_INTERNAL_ERROR("Failed to get DML tensor type.");
  }
  ::dml::TensorDimensions dml_tensor_dims;
  if (!GetDmlTensorDimensions(
      desc->dimensions, desc->dimensionsCount, dml_tensor_dims)) {
    return DAWN_INTERNAL_ERROR("Failed to get DML tensor dimensions.");
  }
  ::dml::TensorDesc dml_tensor_desc(
      dml_tensor_type, dml_tensor_dims, ::dml::TensorPolicy::Default());
  ::dml::Expression dml_input =
      ::dml::InputTensor(*graph_, bindings_.size(), dml_tensor_desc);
  expressions_.insert(std::make_pair(input, dml_input));
  std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(
      dml_input, nullptr, 0));
  bindings_.push_back(std::move(binding));
  inputs_.insert(std::make_pair(input->GetName(), bindings_.back().get()));
  DAWN_DEBUG() << " impl: " << dml_input.Impl()
               << ", name: " << input->GetName()
               << ", type: "
               << DmlTensorDataTypeToString(
                  dml_input.GetOutputDesc().dataType)
               << ", dimensions: "
               << DmlTensorDimensionsToString(
                  dml_input.GetOutputDesc().sizes);
  return {};
}

MaybeError Model::AddOutput(const std::string& name, const OperandBase* output) {
  DAWN_ASSERT(expressions_.find(output) != expressions_.end());
  ::dml::Expression dml_output = expressions_.at(output);
  outputs_.insert(std::make_pair(name, dml_output));
  DAWN_DEBUG() << " impl: " << dml_output.Impl()
               << ", name: " << name;
  return {};
}
  
MaybeError Model::AddBinary(const op::Binary *binary) {
  DAWN_ASSERT(binary->Inputs().size() == 2);
  DAWN_ASSERT(
      expressions_.find(binary->Inputs()[0].Get()) != expressions_.end());
  ::dml::Expression a = expressions_.at(binary->Inputs()[0].Get());
  DAWN_ASSERT(
      expressions_.find(binary->Inputs()[1].Get()) != expressions_.end());
  ::dml::Expression b = expressions_.at(binary->Inputs()[1].Get());
  ::dml::Expression c;
  ::dml::TensorDimensions a_dims = a.GetOutputDesc().sizes;
  const size_t a_rank = a_dims.size();
  ::dml::TensorDimensions b_dims = b.GetOutputDesc().sizes;
  const size_t b_rank = b_dims.size();
  ::dml::TensorDimensions a_new_dims, b_new_dims;
  ::dml::TensorDimensions a_new_strides, b_new_strides;
  bool a_dims_changed = false, b_dims_changed = false;
  size_t c_rank = 0;
  bool need_broadcast = false;
  size_t broadcast_skip_axis = 0;

  if (binary->GetType() == op::BinaryOpType::kMatMul) {
    // DML GEMM requires inputs are either 4D or 5D. We use 4D.
    if (a_rank > 4 || b_rank > 4) {
      return DAWN_INTERNAL_ERROR(
          "The size of input dimensions is greater than 4.");
    }

    if (a_rank == 1 && b_rank == 1) {
      // If both a and b are 1-D, the operation is a vector dot-product,
      // which produces a scalar output.
      c_rank = 1;
    } else {
      // The output is a N-D tensor whose rank is the maximum rank of the
      // input tensors.
      c_rank = std::max(a_rank, b_rank);
    }

    if (a_rank < 4) {
      a_dims = ExpandDimensions(a_dims, 4);
      a_dims_changed = true;
      a_new_dims = a_dims;
      a_new_strides = CalculateStrides(a_new_dims);
    }

    if (b_rank < 4) {
      if (b_rank == 1) {
        // If b is 1-D, it is converted to a 2-D tensor by by appending a 1 to
        // its dimensions.
        b_dims.push_back(1);
      }
      b_dims = ExpandDimensions(b_dims, 4);
      b_dims_changed = true;
      b_new_dims = b_dims;
      b_new_strides = CalculateStrides(b_new_dims);
    }

    if (a_rank > 2 || b_rank > 2) {
      // If either a or b is N-D, N > 2, it is treated as a stack of matrices
      // with dimensions corresponding to the last two indices. The matrix
      // multiplication will be broadcasted accordingly by following
      // [numpy-broadcasting-rule].
      need_broadcast = true;
      broadcast_skip_axis = 2;
    }
  } else {
    // The element-wise binary operation will be broadcasted according to
    // [numpy-broadcasting-rule].
    need_broadcast = true;
    broadcast_skip_axis = 0;
  }

  if (need_broadcast) {
    if (!BroadcastDimensions(
        a_dims, b_dims, a_dims_changed, a_new_dims, a_new_strides,
        b_dims_changed, b_new_dims, b_new_strides, broadcast_skip_axis)) {
      return DAWN_INTERNAL_ERROR("Failed to broadcast a and b.");
    }
  }

  if (a_dims_changed) {
    a = ::dml::Reinterpret(a, a_new_dims, a_new_strides);
  }
  if (b_dims_changed) {
    b = ::dml::Reinterpret(b, b_new_dims, b_new_strides);
  }

  if (binary->GetType() == op::BinaryOpType::kMatMul) {
    c = ::dml::Gemm(a, b);
  } else if (binary->GetType() == op::BinaryOpType::kAdd) {
    c = ::dml::Add(a, b);
  } else if (binary->GetType() == op::BinaryOpType::kMul) {
    c = ::dml::Multiply(a, b);
  } else {
    std::string error_message = std::string(" Binary op ") +
        OpTypeToString(binary->GetType()) +
        std::string(" is not implemented.");
    return DAWN_UNIMPLEMENTED_ERROR(error_message);
  }

  // Reshape back according to c rank if needed.
  ::dml::TensorDimensions c_dims = c.GetOutputDesc().sizes;
  if (c_rank != 0 && c_rank < c_dims.size()) {
    ::dml::TensorDimensions c_new_dims = ShrinkDimensions(c_dims, c_rank);
    ::dml::TensorDimensions c_new_strides = CalculateStrides(c_new_dims);
    c = ::dml::Reinterpret(c, c_new_dims, c_new_strides);
  }
  expressions_.insert(std::make_pair(binary, c));
  DAWN_DEBUG() << " op: " << OpTypeToString(binary->GetType())
               << ", a: {impl: " << a.Impl()
               << ", type: "
               << DmlTensorDataTypeToString(a.GetOutputDesc().dataType)
               << ", dimensions: "
               << DmlTensorDimensionsToString(a.GetOutputDesc().sizes)
               << "}, b: {impl: " << b.Impl()
               << ", type: "
               << DmlTensorDataTypeToString(b.GetOutputDesc().dataType)
               << ", dimensions: "
               << DmlTensorDimensionsToString(b.GetOutputDesc().sizes)
               << "}, c: {impl: " << c.Impl()
               << ", type: "
               << DmlTensorDataTypeToString(c.GetOutputDesc().dataType)
               << ", dimensions: "
               << DmlTensorDimensionsToString(c.GetOutputDesc().sizes)
               << "}";
  return {};
}

MaybeError Model::AddConv2d(const op::Conv2d *conv2d) {
  DAWN_ASSERT(conv2d->Inputs().size() == 2);
  const OperandBase* input_operand = conv2d->Inputs()[0].Get();
  DAWN_ASSERT(expressions_.find(input_operand) != expressions_.end());
  ::dml::Expression input = expressions_.at(input_operand);
  const OperandBase* filter_operand = conv2d->Inputs()[1].Get();
  DAWN_ASSERT(expressions_.find(filter_operand) != expressions_.end());
  ::dml::Expression filter = expressions_.at(filter_operand);
  const Conv2dOptions* options = conv2d->GetOptions();
  ::dml::Expression conv = ::dml::Convolution(
      input, filter, ::dml::NullOpt, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
      DML_CONVOLUTION_DIRECTION_FORWARD,
      // FIXME(nhu): strides, dilations, padding should be uint32_t
      // need to fix the spec.
      // strides
      {
        (const uint32_t)options->strides[0],
        (const uint32_t)options->strides[1]
      },
      // dilations
      {
        (const uint32_t)options->dilations[0],
        (const uint32_t)options->dilations[1]
      },
      // startPadding
      {
        (const uint32_t)options->padding[0],
        (const uint32_t)options->padding[2]
      },
      // endPadding
      {
        (const uint32_t)options->padding[1],
        (const uint32_t)options->padding[3]
      },
      // outPadding
      {},
      // groupCount
      options->groups);
  expressions_.insert(std::make_pair(conv2d, conv));
  return {};
}

MaybeError Model::AddPool2d(const op::Pool2d *pool2d) {
  DAWN_ASSERT(pool2d->Inputs().size() == 1);
  const OperandBase* input_operand = pool2d->Inputs()[0].Get();
  DAWN_ASSERT(expressions_.find(input_operand) != expressions_.end());
  ::dml::Expression input = expressions_.at(input_operand);
  const Pool2dOptions* options = pool2d->GetOptions();
  ::dml::Span<const uint32_t> strides = {
    static_cast<uint32_t>(options->strides[0]),
    static_cast<uint32_t>(options->strides[1])};
  ::dml::Span<const uint32_t> windowSizes = {
    static_cast<uint32_t>(options->windowDimensions[0]),
    static_cast<uint32_t>(options->windowDimensions[1])};
  ::dml::Span<const uint32_t> startPadding {
    static_cast<uint32_t>(options->padding[0]),
    static_cast<uint32_t>(options->padding[2])};
  ::dml::Span<const uint32_t> endPadding = {
    static_cast<uint32_t>(options->padding[1]),
    static_cast<uint32_t>(options->padding[3])};
  ::dml::Span<const uint32_t> dilations = {
    static_cast<uint32_t>(options->dilations[0]),
    static_cast<uint32_t>(options->dilations[1])};
  ::dml::Expression output;
  if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
    DAWN_ASSERT(dilations[0] == 1 || dilations[1] == 1);
    output = ::dml::AveragePooling(
        input, strides, windowSizes, startPadding, endPadding, false);
  } else if (pool2d->GetType() == op::Pool2dType::kMaxPool2d) {
    output = ::dml::MaxPooling(
        input, windowSizes, strides, startPadding, endPadding, dilations,
        false).values;
  } else {
    UNREACHABLE();
  }
  expressions_.insert(std::make_pair(pool2d, output));
  return {};
}

MaybeError Model::AddReshape(const op::Reshape *reshape) {
  DAWN_ASSERT(reshape->Inputs().size() == 1);
  const OperandBase* input_operand = reshape->Inputs()[0].Get();
  DAWN_ASSERT(expressions_.find(input_operand) != expressions_.end());
  ::dml::Expression input = expressions_.at(input_operand);
  ::dml::TensorDimensions new_sizes;
  DAWN_ASSERT(reshape->GetNewShapeCount() <= 4);
  new_sizes.assign(reshape->GetNewShape(),
                   reshape->GetNewShape() + reshape->GetNewShapeCount());
  ::dml::Expression output =
      ::dml::Reinterpret(input, new_sizes, ::dml::NullOpt);
  expressions_.insert(std::make_pair(reshape, output));
  return {};
}

MaybeError Model::AddTranspose(const op::Transpose *transpose) {
  return DAWN_UNIMPLEMENTED_ERROR("Transpose");
}
  
MaybeError Model::AddUnary(const op::Unary *unary) {
  DAWN_ASSERT(unary->Inputs().size() == 1);
  const OperandBase* input_operand = unary->Inputs()[0].Get();
  DAWN_ASSERT(expressions_.find(input_operand) != expressions_.end());
  ::dml::Expression input = expressions_.at(input_operand);
  ::dml::Expression output;
  if (unary->GetType() == op::UnaryOpType::kRelu) {
    output = ::dml::ActivationRelu(input);
  } else if (unary->GetType() == op::UnaryOpType::kSoftmax) {
    output = ::dml::ActivationSoftmax(input);
  } else {
    std::string error_message = std::string(" Unary op ") +
        OpTypeToString(unary->GetType()) + std::string(" is not implemented.");
    return DAWN_UNIMPLEMENTED_ERROR(error_message);
  }
  expressions_.insert(std::make_pair(unary, output));
  DAWN_DEBUG() << " op: " << OpTypeToString(unary->GetType())
               << ", input: {impl: " << input.Impl()
               << ", type: "
               << DmlTensorDataTypeToString(input.GetOutputDesc().dataType)
               << ", dimensions: "
               << DmlTensorDimensionsToString(input.GetOutputDesc().sizes)
               << "}, output: {impl: " << output.Impl()
               << ", type: "
               << DmlTensorDataTypeToString(output.GetOutputDesc().dataType)
               << ", dimensions: "
               << DmlTensorDimensionsToString(output.GetOutputDesc().sizes)
               << "}";
  return {};
}

MaybeError Model::Finish() {
  if (outputs_.size() == 1) {
    auto output = outputs_.begin();
    if (output->second.Impl()->GetNode().type ==
        ::dml::detail::NodeType::Reinterpret) {
      // Deal with a graph with single reshape node.
      // https://github.com/microsoft/DirectML/issues/71
      std::string name = output->first;
      ::dml::Expression reshape = output->second;
      outputs_[name] = ::dml::ActivationIdentity(reshape);
    }
  }
  return {};
}

void Model::CompileImpl(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  // FIXME(nhu): implement async
  WNNCompileStatus status = WNNCompileStatus_Success;
  callback(status, reinterpret_cast<WNNCompilation>(new Compilation(this)),
           nullptr, userdata);
}

}  // namespace dml
}  // namespace dawn_native
