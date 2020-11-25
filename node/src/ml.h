#ifndef __ML_H__
#define __ML_H__

#include "Base.h"

class ML : public Napi::ObjectWrap<ML> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  ML(const Napi::CallbackInfo &info);
  ~ML();
private:
  static Napi::Value GetNeuralNetworkContext(const Napi::CallbackInfo &info);
};

#endif // __ML_H__
