
#include "dawn_native/ModelBuilder.h"

#include <string>

#include "common/Assert.h"
#include "common/RefCounted.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ops/binary.h"
#include "dawn_native/ops/constant.h"
#include "dawn_native/ops/conv2d.h"
#include "dawn_native/ops/input.h"
#include "dawn_native/ops/pool2d.h"
#include "dawn_native/ops/reshape.h"
#include "dawn_native/ops/transpose.h"
#include "dawn_native/ops/unary.h"

namespace dawn_native {

OperandBase *ModelBuilderBase::Constant(OperandDescriptor const *desc,
                                        void const *value, size_t size) {
  Ref<OperandBase> context = AcquireRef(new op::Constant(desc, value, size));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Input(char const *name,
                                     OperandDescriptor const *desc) {
  Ref<OperandBase> context = AcquireRef(new op::Input(std::string(name), desc));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Matmul(OperandBase *a, OperandBase *b) {
   Ref<OperandBase> context =
      AcquireRef(new op::Binary(op::BinaryOpType::kMatMul, a, b));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Add(OperandBase *a, OperandBase *b) {
  Ref<OperandBase> context =
      AcquireRef(new op::Binary(op::BinaryOpType::kAdd, a, b));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Mul(OperandBase *a, OperandBase *b) {
  Ref<OperandBase> context =
      AcquireRef(new op::Binary(op::BinaryOpType::kMul, a, b));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Conv2d(OperandBase *input, OperandBase *filter,
                                      Conv2dOptions const *options) {
  Ref<OperandBase> context = AcquireRef(new op::Conv2d(input, filter, options));
  return context.Detach();
}

OperandBase *ModelBuilderBase::AveragePool2d(OperandBase *input,
                                             Pool2dOptions const *options) {
  Ref<OperandBase> context = AcquireRef(
      new op::Pool2d(op::Pool2dType::kAveragePool2d, input, options));
  return context.Detach();
}

OperandBase *ModelBuilderBase::MaxPool2d(OperandBase *input,
                                         Pool2dOptions const *options) {
  Ref<OperandBase> context =
      AcquireRef(new op::Pool2d(op::Pool2dType::kMaxPool2d, input, options));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Relu(OperandBase *input) {
  Ref<OperandBase> context =
      AcquireRef(new op::Unary(op::UnaryOpType::kRelu, input));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Reshape(OperandBase *input,
                                       int32_t const *new_shape,
                                       size_t new_shape_count) {
  Ref<OperandBase> context =
      AcquireRef(new op::Reshape(input, new_shape, new_shape_count));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Softmax(OperandBase *input) {
  Ref<OperandBase> context =
      AcquireRef(new op::Unary(op::UnaryOpType::kSoftmax, input));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Transpose(OperandBase *input,
                                         TransposeOptions const *options) {
  Ref<OperandBase> context = AcquireRef(new op::Transpose(input, options));
  return context.Detach();
}

ModelBase *ModelBuilderBase::CreateModel(NamedOperandsBase const *named_operands) {
  return CreateModelImpl(named_operands);
}

} // namespace dawn_native
