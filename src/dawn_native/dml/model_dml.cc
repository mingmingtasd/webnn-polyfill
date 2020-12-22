#include "dawn_native/dml/model_dml.h"

#include "common/Assert.h"
#include "common/Log.h"
#include "dawn_native/dml/deps/src/precomp.h"
#include "dawn_native/dml/compilation_dml.h"

namespace dawn_native {
namespace dml {

namespace {
DML_TENSOR_DATA_TYPE getDmlTensorDataType(wnn::OperandType operand_type) {
  if (operand_type == wnn::OperandType::Float32) {
    return DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if (operand_type == wnn::OperandType::Float16) {
    return DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if (operand_type == wnn::OperandType::Int32) {
    return DML_TENSOR_DATA_TYPE_INT32;
  } else if (operand_type == wnn::OperandType::Uint32) {
    return DML_TENSOR_DATA_TYPE_UINT32;
  }
  return DML_TENSOR_DATA_TYPE_UNKNOWN;
}

::dml::TensorDimensions getDmlTensorDimensions(
    int32_t const * dimensions, uint32_t dimensions_count) {
  // DML dimension order [N, C, H, W]
  const size_t dml_dimensions_count = 4;
  ::dml::TensorDimensions tensor_dimensions({1, 1, 1, 1});
  DAWN_ASSERT(dimensions_count <= 4);
  for (uint32_t i = 0; i < dimensions_count; ++i) {
    tensor_dimensions[dml_dimensions_count - i - 1] =
        dimensions[dimensions_count - i - 1];
  }
  return tensor_dimensions;
}
}  // namespace

Model::Model() {
  device_.reset(new ::pydml::Device());
  graph_.reset(new ::dml::Graph(device_->GetDevice()));
}

void Model::AddConstant(const op::Constant *constant) {
  const OperandDescriptor* desc = constant->GetOperandDescriptor();
  ::dml::TensorDimensions dml_dims =
      getDmlTensorDimensions(desc->dimensions, desc->dimensionsCount);
  ::dml::TensorDesc tensor_desc(
      getDmlTensorDataType(desc->type),
      ::DML_TENSOR_FLAGS::DML_TENSOR_FLAG_OWNED_BY_DML,
      dml_dims,
      ::dml::TensorPolicy::Default());
  ::dml::Expression exp =
      ::dml::InputTensor(*graph_, bindings_.size(), tensor_desc);
  expressions_.insert(std::make_pair(constant, exp));
  std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(
      exp, const_cast<void*>(constant->GetValue()), constant->GetSize()));
  bindings_.push_back(std::move(binding));
  DAWN_DEBUG();
}

void Model::AddInput(const op::Input *input) {
  const OperandDescriptor* desc = input->GetOperandDescriptor();
  ::dml::TensorDimensions dml_dims =
      getDmlTensorDimensions(desc->dimensions, desc->dimensionsCount);
  ::dml::TensorDesc tensor_desc(
      getDmlTensorDataType(desc->type),
      dml_dims,
      ::dml::TensorPolicy::Default());
  ::dml::Expression exp =
      ::dml::InputTensor(*graph_, bindings_.size(), tensor_desc);
  expressions_.insert(std::make_pair(input, exp));
  std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(
      exp, nullptr, 0));
  bindings_.push_back(std::move(binding));
  inputs_.insert(std::make_pair(input->GetName(), bindings_.back().get()));
  DAWN_DEBUG() << " " << input->GetName();
}

void Model::AddOutput(const std::string& name, const OperandBase* output) {
  DAWN_ASSERT(expressions_.find(output) != expressions_.end());
  outputs_.insert(std::make_pair(name, expressions_.at(output)));
  DAWN_DEBUG() << " " << name;
}
  
void Model::AddBinary(const op::Binary *binary) {
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
    c = ::dml::Gemm(a, b);
  } else {
    UNREACHABLE();
  }
  expressions_.insert(std::make_pair(binary, c));
}

void Model::AddConv2d(const op::Conv2d *conv2d) {
  DAWN_ASSERT(conv2d->Inputs().size() == 2);
  const OperandBase* input_operand = conv2d->Inputs()[0].Get();
  DAWN_ASSERT(expressions_.find(input_operand) != expressions_.end());
  ::dml::Expression input = expressions_.at(input_operand);
  const OperandBase* filter_operand = conv2d->Inputs()[1].Get();
  DAWN_ASSERT(expressions_.find(filter_operand) != expressions_.end());
  ::dml::Expression filter = expressions_.at(filter_operand);
  const Conv2dOptions* options = conv2d->Options();
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
}

void Model::AddPool2d(const op::Pool2d *pool2d) {
  DAWN_ASSERT(pool2d->Inputs().size() == 1);
  const OperandBase* input_operand = pool2d->Inputs()[0].Get();
  DAWN_ASSERT(expressions_.find(input_operand) != expressions_.end());
  ::dml::Expression input = expressions_.at(input_operand);
  const Pool2dOptions* options = pool2d->Options();
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
}

void Model::AddReshape(const op::Reshape *relu) {
  UNREACHABLE();
}

void Model::AddTranspose(const op::Transpose *transpose) {
  UNREACHABLE();
}
  
void Model::AddUnary(const op::Unary *unary) {
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
    UNREACHABLE();
  }
  expressions_.insert(std::make_pair(unary, output));
}

void Model::Finish() {
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
}

void Model::CompileImpl(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  // FIXME(nhu): implement async
  callback(reinterpret_cast<WNNCompilation>(new Compilation(this)), userdata);
}

}  // namespace dml
}  // namespace dawn_native
