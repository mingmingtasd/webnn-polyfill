#ifndef __OPERAND_H__
#define __OPERAND_H__

#include "ops/Node.h"

class Operand : public Napi::ObjectWrap<Operand> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  explicit Operand(const Napi::CallbackInfo &info);
  ~Operand() = default;

  void SetNode(std::shared_ptr<op::Node>);
  WebnnOperand GetOperand();

private:
  std::shared_ptr<op::Node> node_;
};

#endif // __OPERAND_H__
