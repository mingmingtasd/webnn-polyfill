#ifndef __NEURAL_NETWORK_CONTEXT_H__
#define __NEURAL_NETWORK_CONTEXT_H__

#include "Base.h"

class NeuralNetworkContext : public Napi::ObjectWrap<NeuralNetworkContext> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  NeuralNetworkContext(const Napi::CallbackInfo &info);
  ~NeuralNetworkContext();

  // #accessors
  Napi::Value CreateModelBuilder(const Napi::CallbackInfo &info);

  WebnnNeuralNetworkContext GetContext();

private:
  WebnnNeuralNetworkContext context_;
};

#endif // __NEURAL_NETWORK_CONTEXT_H__
