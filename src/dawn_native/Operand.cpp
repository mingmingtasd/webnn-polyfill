
#include "dawn_native/Operand.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {

Ref<OperandBase> OperandBase::FirstInput() const { return nullptr; }

Ref<OperandBase> OperandBase::NextInput() { return nullptr; }

void OperandBase::AddOperand(ModelBase *model) { UNREACHABLE(); }

void OperandBase::SetTraversal(bool traversal) { traversalled_ = traversal; }

bool OperandBase::Traversal() { return traversalled_; }

void OperandBase::SetName(std::string name) { name_ = name; }

std::string OperandBase::GetName() { return name_; }

} // namespace dawn_native
