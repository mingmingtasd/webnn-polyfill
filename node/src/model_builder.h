#ifndef __MODEL_BUILDER_H__
#define __MODEL_BUILDER_H__

#include "Base.h"

template <typename T>
Napi::Value AddOperandToModel(const Napi::CallbackInfo &info,
                              WNNModelBuilder builder) {
  Napi::Object object = Operand::constructor.New({});
  Operand* unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto operand = std::make_shared<T>(info);
  operand->AddToModel(builder);
  unwrapped->SetOperand(operand);

  return object;
}

class ModelBuilder : public Napi::ObjectWrap<ModelBuilder> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  explicit ModelBuilder(const Napi::CallbackInfo &info);
  ~ModelBuilder();

  // #accessors
  Napi::Value Constant(const Napi::CallbackInfo &info);
  Napi::Value Input(const Napi::CallbackInfo &info);
  Napi::Value Add(const Napi::CallbackInfo &info);
  Napi::Value MatMul(const Napi::CallbackInfo &info);
  Napi::Value CreateModel(const Napi::CallbackInfo &info);

private:
  WNNModelBuilder model_builder_;
};

#endif // __MODEL_BUILDER_H__
