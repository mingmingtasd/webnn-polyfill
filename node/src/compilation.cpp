#include "compilation.h"

#include <iostream>

#include "DescriptorDecoder.h"
#include "model.h"

Napi::Object OutputItem(const Napi::Env &env, WNNResult result) {
  Napi::Object item = Napi::Object::New(env);
  size_t buffer_size = wnnResultBufferSize(result) / sizeof(float);
  Napi::Float32Array output_array = Napi::Float32Array::New(env, buffer_size);
  const float *buffer = static_cast<const float *>(wnnResultBuffer(result));
  for (size_t i = 0; i < buffer_size; ++i) {
    output_array[i] = buffer[i];
  }
  item.Set("buffer", output_array);
  return item;
}

Napi::FunctionReference Compilation::constructor;

Compilation::Compilation(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<Compilation>(info) {
  this->model_object_.Reset(info[0].ToObject(), 1);
}

Compilation::~Compilation() {
  this->model_object_.Reset();
  wnnCompilationRelease(compilation_);
}

void Compilation::SetCompilation(WNNCompilation Compilation) {
  compilation_ = Compilation;
}

WNNCompilation Compilation::GetCompilation() {
  return compilation_;
}

void Compilation::SetNamedResults(WNNNamedResults named_results) {
  named_results_ = named_results;
}

Napi::Value Compilation::Compute(const Napi::CallbackInfo &info) {
  WNNNamedInputs named_inputs = dawn_native::CreateNamedInputs();
  // The WNNInput struct need to be kept until compute.
  std::vector<WNNInput> inputs;
  if (info[0].IsObject()) {
    Napi::Object obj = info[0].As<Napi::Object>();
    Napi::Array property_names = obj.GetPropertyNames();
    for (size_t j = 0; j < property_names.Length(); ++j) {
      std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
      inputs.push_back(ParseOperand<WNNInput>(obj, name));
      wnnNamedInputsSet(named_inputs, name.data(), &inputs[j]);
    }
  }

  // The WNNOutput struct need to be kept until compute.
  std::vector<WNNOutput> outputs;
  WNNNamedOutputs named_outputs = nullptr;
  if (info.Length() > 1) {
    named_outputs = dawn_native::CreateNamedOutputs();
    if (info[1].IsObject()) {
      Napi::Object obj = info[1].As<Napi::Object>();
      Napi::Array property_names = obj.GetPropertyNames();
      for (size_t j = 0; j < property_names.Length(); ++j) {
        std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
        outputs.push_back(ParseOperand<WNNOutput>(obj, name));
        wnnNamedOutputsSet(named_outputs, name.data(), &outputs[j]);
      }
    }
  }

  wnnCompilationCompute(
      compilation_, named_inputs,
      [](WNNNamedResults results, void *user_data) {
        reinterpret_cast<Compilation *>(user_data)->SetNamedResults(results);
      },
      reinterpret_cast<void *>(this), named_outputs);

  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  if (info.Length() == 1) {
    Napi::Object obj = Napi::Object::New(env);
    Model *model = Napi::ObjectWrap<Model>::Unwrap(model_object_.Value());
    for (auto &name : model->GetOutputName()) {
      WNNResult result = wnnNamedResultsGet(named_results_, name.data());
      obj.Set(name, OutputItem(env, result));
      wnnResultRelease(result);
    }
    // Free native memory.
    wnnNamedResultsRelease(named_results_);
    deferred.Resolve(obj);
  } else {
    deferred.Resolve(info[1]);
  }
  return deferred.Promise();
}

Napi::Object Compilation::Initialize(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  Napi::Function func = DefineClass(env, "Compilation", {
    InstanceMethod(
      "compute",
      &Compilation::Compute,
      napi_enumerable
    )
  });
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Compilation", func);
  return exports;
}
