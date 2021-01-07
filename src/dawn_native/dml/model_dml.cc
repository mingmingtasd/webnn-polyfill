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
  if (binary->GetType() == op::BinaryOpType::kAdd) {
    c = ::dml::Add(a, b);
  } else if (binary->GetType() == op::BinaryOpType::kMul) {
    c = ::dml::Multiply(a, b);
  } else if (binary->GetType() == op::BinaryOpType::kMatMul) {
    // GEMM requires inputs are 4D.
    c = ::dml::Gemm(a, b);
  } else {
    std::string error_message = std::string(" Binary op ") +
        OpTypeToString(binary->GetType()) + std::string(" is not implemented.");
    return DAWN_UNIMPLEMENTED_ERROR(error_message);
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
  size_t op_count = expressions_.size() - bindings_.size();
  DAWN_DEBUG() << " op count: " << op_count;
  // FIXME(nhu): workaround the optional tensor issue of DML
  // https://github.com/microsoft/DirectML/issues/64
  if (op_count == 1) {
    for (auto& output : outputs_) {
      DAWN_DEBUG() << " append an activation identity for output "
                   << output.first;
      ::dml::Expression identity = ::dml::ActivationIdentity(output.second);
      outputs_[output.first] = identity;
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
