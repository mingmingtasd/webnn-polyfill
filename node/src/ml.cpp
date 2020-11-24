#include "ML.h"

#include "neural_network_context.h"

Napi::FunctionReference ML::constructor;

ML::ML(const Napi::CallbackInfo& info) : Napi::ObjectWrap<ML>(info) { }
ML::~ML() { }

Napi::Value ML::GetNeuralNetworkContext(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Object context = NeuralNetworkContext::constructor.New({
      info.This().As<Napi::Value>()});

  return context;
}

Napi::Object ML::Initialize(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  Napi::Function func = DefineClass(env, "ML", {
    StaticMethod(
      "getNeuralNetworkContext",
      &ML::GetNeuralNetworkContext,
      napi_enumerable
    )
  });
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("ML", func);
  return exports;
}
