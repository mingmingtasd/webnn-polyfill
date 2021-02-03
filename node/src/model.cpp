#include "model.h"

#include <iostream>

#include "compilation.h"

Napi::FunctionReference Model::constructor;

Model::Model(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Model>(info) {
  Napi::Object obj = info[0].As<Napi::Object>();
  Napi::Array property_names = obj.GetPropertyNames();
  for (size_t j = 0; j < property_names.Length(); ++j) {
    std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
    output_name_.push_back(name);
  }
}

Model::~Model() {
  webnnModelRelease(model_);
}

void Model::SetModel(WebnnModel model) {
  model_ = model;
}

WebnnModel Model::GetModel() {
  return model_;
}

void Model::SetWebnnCompilation(WebnnCompilation compilation) {
  compilation_ = compilation;
}

std::vector<std::string> &Model::GetOutputName() { return output_name_; }

Napi::Value Model::Compile(const Napi::CallbackInfo &info) {
  webnnModelCompile(
      model_,
      [](WebnnCompileStatus status, WebnnCompilation compilation,
         char const *message, void *userData) {
        Model *self = reinterpret_cast<Model *>(userData);
        self->SetWebnnCompilation(compilation);
      },
      reinterpret_cast<void *>(this), nullptr);

  Napi::Env env = info.Env();
  std::vector<napi_value> args = {info.This().As<Napi::Value>()};
  Napi::Object compilation = Compilation::constructor.New(args);
  Compilation* unwrapped = Napi::ObjectWrap<Compilation>::Unwrap(compilation);
  unwrapped->SetCompilation(compilation_);

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
