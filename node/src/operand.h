#ifndef __OPERAND_H__
#define __OPERAND_H__

#include "ops/operand_wrap.h"

class Operand : public Napi::ObjectWrap<Operand> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  explicit Operand(const Napi::CallbackInfo &info);
  ~Operand() = default;

  void SetOperand(std::shared_ptr<op::OperandWrap>);
  WNNOperand GetOperand();

private:
  std::shared_ptr<op::OperandWrap> operand_wrap_;
};

#endif // __OPERAND_H__
