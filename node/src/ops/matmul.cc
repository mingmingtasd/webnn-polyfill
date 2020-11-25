#include "MatMul.h"

#include "../operand.h"

namespace op {

MatMul::MatMul(const Napi::CallbackInfo &info) {
  Operand* primary = Napi::ObjectWrap<Operand>::Unwrap(info[0].As<Napi::Object>());
  primary_ = primary->GetOperand();
  Operand* secondary = Napi::ObjectWrap<Operand>::Unwrap(info[1].As<Napi::Object>());
  secondary_ = secondary->GetOperand();
}

void MatMul::AddToModel(WNNModelBuilder builder) {
  OperandWrap::SetOperand(wnnModelBuilderMatmul(builder,
                                              primary_,
                                              secondary_));
}

} // namespace op
