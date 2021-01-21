#include "dawn_native/Operand.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {

OperandBase::OperandBase(ModelBuilderBase *model_builder,
                         std::vector<Ref<OperandBase>> inputs)
    : ObjectBase(model_builder->GetContext()), inputs_(std::move(inputs)) {}

OperandBase::OperandBase(ModelBuilderBase *model_builder,
                         ObjectBase::ErrorTag tag)
    : ObjectBase(model_builder->GetContext(), tag) {}

// static
OperandBase *OperandBase::MakeError(ModelBuilderBase *model_builder) {
  return new OperandBase(model_builder, ObjectBase::kError);
}

MaybeError OperandBase::AddToModel(ModelBase *model) const { UNREACHABLE(); }

const std::vector<Ref<OperandBase>> &OperandBase::Inputs() const { return inputs_; }

} // namespace dawn_native
