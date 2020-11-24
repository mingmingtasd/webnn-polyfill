#include "operand_wrap.h"

namespace op {

void OperandWrap::SetOperand(WNNOperand operand) {
  wnn_operand_ = operand;
}

WNNOperand OperandWrap::GetOperand() {
  return wnn_operand_;
}

OperandWrap::~OperandWrap() {
  wnnOperandRelease(wnn_operand_);
}

} // namespace op
