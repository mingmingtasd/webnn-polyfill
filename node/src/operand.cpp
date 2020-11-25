#include "operand.h"

Napi::FunctionReference Operand::constructor;

Operand::Operand(const Napi::CallbackInfo& info) : 
    Napi::ObjectWrap<Operand>(info) {
}
void Operand::SetNode(std::shared_ptr<op::Node> node) { node_ = node; }

WNNOperand Operand::GetOperand() { return node_->GetOutput(); }

Napi::Object Operand::Initialize(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  Napi::Function func = DefineClass(env, "Operand", {
  });
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Operand", func);
  return exports;
}
