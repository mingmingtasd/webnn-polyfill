#include "model.h"

#include <iostream>

#include "DescriptorDecoder.h"
#include "compilation.h"

Napi::FunctionReference Model::constructor;

Model::Model(const Napi::CallbackInfo& info) : 
    Napi::ObjectWrap<Model>(info) {}

Model::~Model() {
  wnnModelRelease(model_);
}

void Model::SetModel(WNNModel model) {
  model_ = model;
}

WNNModel Model::GetModel() {
  return model_;
}

void Model::SetWNNCompilation(WNNCompilation compilation) {
  compilation_ = compilation;
}

Napi::Value Model::Compile(const Napi::CallbackInfo &info) {
  wnnModelCompile(model_, [](WNNCompilation compilation, void* userData){
          Model* self = reinterpret_cast<Model*>(userData);
          self->SetWNNCompilation(compilation);
        }, reinterpret_cast<void*>(this), nullptr);
  Napi::Object compilation = Compilation::constructor.New({});
  Compilation* unwrapped = Napi::ObjectWrap<Compilation>::Unwrap(compilation);
  unwrapped->SetCompilation(compilation_);

  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  deferred.Resolve(compilation);
  return deferred.Promise();
}

Napi::Object Model::Initialize(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  Napi::Function func = DefineClass(env, "Model", {
    InstanceMethod(
      "compile",
      &Model::Compile,
      napi_enumerable
    )
  });
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Model", func);
  return exports;
}
