#ifndef __OPS_OPERAND_H_
#define __OPS_OPERAND_H_

#include <string>
#include <vector>

#include "../Base.h"

namespace op {

class OperandWrap {
public:
  OperandWrap() = default;
  ~OperandWrap();

  void SetOperand(WNNOperand);
  WNNOperand GetOperand();

private:
  WNNOperand wnn_operand_;
};
} // op

#endif  // __OPS_OPERAND_H_