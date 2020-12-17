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
    int32_t const * dimensions, uint32_t dimensionsCount) {
  // DML dimension order [N, C, H, W]
  ::dml::TensorDimensions tensor_dimensions({1, 1, 1, 1});
  DAWN_ASSERT(dimensionsCount <= 4);
  for (uint32_t i = 0; i < dimensionsCount; ++i) {
    tensor_dimensions[3 - i] = dimensions[i];
  }
  return tensor_dimensions;
}
}  // namespace

Model::Model() : input_index_(0) {
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
      ::dml::InputTensor(*graph_, input_index_++, tensor_desc);
  expressions_.insert(std::make_pair(constant, exp));
  std::unique_ptr<::pydml::Binding> binding(new ::pydml::Binding(
      exp, const_cast<void*>(constant->GetValue()), constant->GetSize()));
  bindings_.push_back(std::move(binding));
}

void Model::AddInput(const op::Input *input) {
  const OperandDescriptor* desc = input->GetOperandDescriptor();
  ::dml::TensorDimensions dml_dims =
      getDmlTensorDimensions(desc->dimensions, desc->dimensionsCount);
  ::dml::TensorDesc tensor_desc(
      getDmlTensorDataType(desc->type),
      ::DML_TENSOR_FLAGS::DML_TENSOR_FLAG_OWNED_BY_DML,
      dml_dims,
      ::dml::TensorPolicy::Default());
  ::dml::Expression exp =
      ::dml::InputTensor(*graph_, input_index_++, tensor_desc);
  expressions_.insert(std::make_pair(input, exp));
  inputs_.insert(std::make_pair(input->GetName(), exp));
}

void Model::AddOutput(const std::string& name, const OperandBase* output) {
  DAWN_ASSERT(expressions_.find(output) != expressions_.end());
  outputs_.insert(std::make_pair(name, expressions_.at(output)));
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
  ::dml::Expression output = ::dml::Convolution(
      input, filter, ::dml::NullOpt, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
      DML_CONVOLUTION_DIRECTION_FORWARD,
      // FIXME(nhu): strides, dilations, padding should be uint32_t
      // need to fix the spec.
      // strides
      {
        reinterpret_cast<const uint32_t*>(options->strides),
        options->stridesCount
      },
      // dilations
      {
        reinterpret_cast<const uint32_t*>(options->dilations),
        options->dilationsCount
      },
      // startPadding
      {
        (const uint32_t)options->padding[0],
        (const uint32_t)options->padding[2]
      },
      // endPadding
      {
        (const uint32_t)options->padding[1],
        (const uint32_t)options->padding[3],
      },
      // outPadding
      {},
      // groupCount
      options->groups);
  expressions_.insert(std::make_pair(conv2d, output));
}
  
void Model::AddPool2d(const op::Pool2d *pool2d) {
  UNREACHABLE();
}

void Model::AddReshape(const op::Reshape *relu) {
  UNREACHABLE();
}

void Model::AddTranspose(const op::Transpose *transpose) {
  UNREACHABLE();
}
  
void Model::AddUnary(const op::Unary *unary) {
  UNREACHABLE();
}

void Model::Finish() {}

void Model::CompileImpl(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  // FIXME(nhu): implement async
  callback(reinterpret_cast<WNNCompilation>(new Compilation(this)), userdata);
}

}  // namespace dml
}  // namespace dawn_native
