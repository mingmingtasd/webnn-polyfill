#include "NeuralNetworkContext.h"

#include "ModelBuilder.h"

Napi::FunctionReference NeuralNetworkContext::constructor;

NeuralNetworkContext::NeuralNetworkContext(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<NeuralNetworkContext>(info) {
  WebnnProcTable backendProcs = webnn_native::GetProcs();
  webnnProcSetProcs(&backendProcs);
  context_ = webnn_native::CreateNeuralNetworkContext();
  if (context_ == nullptr) {
      Napi::Env env = info.Env();
      Napi::Error::New(env, "Failed to create neural network context").ThrowAsJavaScriptException();
      return;
  }
}

NeuralNetworkContext::~NeuralNetworkContext() {
  webnnNeuralNetworkContextRelease(context_);
}

WebnnNeuralNetworkContext NeuralNetworkContext::GetContext() { return context_; }

Napi::Value NeuralNetworkContext::CreateModelBuilder(const Napi::CallbackInfo &info) {
  std::vector<napi_value> args = {info.This().As<Napi::Value>()};
  Napi::Object model_builder = ModelBuilder::constructor.New(args);

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
