#include "operand.h"

Napi::FunctionReference Operand::constructor;

Operand::Operand(const Napi::CallbackInfo& info) : 
    Napi::ObjectWrap<Operand>(info) {
}
void Operand::SetOperand(std::shared_ptr<op::OperandWrap> operand) {
  operand_wrap_ = operand;
}

WNNOperand Operand::GetOperand() {
  return operand_wrap_->GetOperand();
}

Napi::Object Operand::Initialize(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  Napi::Function func = DefineClass(env, "Operand", {
  });
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Operand", func);
  return exports;
}
