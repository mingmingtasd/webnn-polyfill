
#include "dawn_native/ModelBuilder.h"

#include <string>

#include "common/Assert.h"
#include "common/RefCounted.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ops/binary.h"
#include "dawn_native/ops/constant.h"
#include "dawn_native/ops/input.h"
#include "dawn_native/ops/matmul.h"

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
  Ref<OperandBase> context = AcquireRef(new op::MatMul(a, b));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Add(OperandBase *a, OperandBase *b) {
  Ref<OperandBase> context =
      AcquireRef(new op::Binary(op::kBinaryTypeAdd, a, b));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Mul(OperandBase *a, OperandBase *b) {
  Ref<OperandBase> context =
      AcquireRef(new op::Binary(op::kBinaryTypeMul, a, b));
  return context.Detach();
}

ModelBase *ModelBuilderBase::CreateModel(NamedOperand const *named_operand,
                                         size_t size) {
  return CreateModelImpl(named_operand, size);
}

} // namespace dawn_native
