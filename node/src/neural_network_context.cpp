#include "neural_network_context.h"

#include "model_builder.h"

Napi::FunctionReference NeuralNetworkContext::constructor;

NeuralNetworkContext::NeuralNetworkContext(const Napi::CallbackInfo& info) : 
    Napi::ObjectWrap<NeuralNetworkContext>(info) {}

NeuralNetworkContext::~NeuralNetworkContext() {}

Napi::Value NeuralNetworkContext::CreateModelBuilder(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Object model_builder = ModelBuilder::constructor.New({});

  return model_builder;
}

Napi::Object NeuralNetworkContext::Initialize(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  Napi::Function func = DefineClass(env, "NeuralNetworkContext", {
    InstanceMethod(
      "createModelBuilder",
      &NeuralNetworkContext::CreateModelBuilder,
      napi_enumerable
    )
  });
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("NeuralNetworkContext", func);
  return exports;
}
