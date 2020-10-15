
#include "dawn_native/ie/operand_ie.h"

namespace dawn_native {

namespace ie {
Operand::Operand(Ref<op::Operand> operand) : operand_(std::move(operand)) {}

Ref<op::Operand> Operand::GetOperand() { return operand_; }
} // namespace ie

} // namespace dawn_native
