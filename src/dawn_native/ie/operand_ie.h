#ifndef WEBNN_NATIVE_IE_OPERAND_IE_H_
#define WEBNN_NATIVE_IE_OPERAND_IE_H_

#include "dawn_native/Operand.h"
#include "dawn_native/ops/operand.h"

namespace dawn_native {

namespace ie {

class Operand : public OperandBase {
public:
  explicit Operand(Ref<op::Operand> operand);
  ~Operand() override = default;

  Ref<op::Operand> GetOperand();

private:
  Ref<op::Operand> operand_;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_OPERAND_IE_H_