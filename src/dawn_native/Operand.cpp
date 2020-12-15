#include "dawn_native/Operand.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {

OperandBase::OperandBase(std::vector<Ref<OperandBase>> inputs)
    : inputs_(std::move(inputs)) {}

void OperandBase::AddToModel(ModelBase *model) const { UNREACHABLE(); }

const std::vector<Ref<OperandBase>> &OperandBase::Inputs() const { return inputs_; }

void OperandBase::SetName(std::string name) { name_ = name; }

} // namespace dawn_native
