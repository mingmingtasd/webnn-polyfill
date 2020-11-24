#include "Compilation.h"

#include <iostream>

#include "DescriptorDecoder.h"

Napi::Object OutputItem(const Napi::Env& env, WNNOutputs outputs, size_t index) {
  Napi::Object item = Napi::Object::New(env);
  WNNOutput output = wnnOutputsGetOutputWithIndex(outputs, index);
  size_t buffer_size = output.size / sizeof(float);
  Napi::Float32Array output_array = Napi::Float32Array::New(env, buffer_size);
  float* buffer = static_cast<float*>(output.buffer);
  for (size_t i = 0; i < buffer_size; ++i) {
    output_array[i] = buffer[i];
  }
  item.Set("buffer", output_array);
  return item;
}

Napi::FunctionReference Compilation::constructor;

Compilation::Compilation(const Napi::CallbackInfo& info) : 
    Napi::ObjectWrap<Compilation>(info) {}

Compilation::~Compilation() {
  FreeUnusedData();
  wnnCompilationRelease(compilation_);
}

void Compilation::SetCompilation(WNNCompilation Compilation) {
  compilation_ = Compilation;
}

WNNCompilation Compilation::GetCompilation() {
  return compilation_;
}

void Compilation::FreeUnusedData() {
  for (auto input : inputs_) {
    delete input;
  }
  inputs_.resize(0);
  for (auto output : outputs_) {
    delete output;
  }
  outputs_.resize(0);
}

Napi::Value Compilation::Compute(const Napi::CallbackInfo &info) {
  FreeUnusedData();

  WNNInputs inputs = dawn_native::CreateInputs();
  if (info[0].IsArray()) {
    Napi::Array input_array = info[0].As<Napi::Array>();
    // [{"input0" : {buffer : inputbuffer0} }]
    for (size_t i = 0; i < input_array.Length(); ++i) {
      Napi::Object item = input_array.Get(i).As<Napi::Object>();
      std::string name;
      WNNInput* input = ParseOperand<WNNInput>(item, name);
      inputs_.push_back(input);
      wnnInputsSetInput(inputs, name.data(), input);
    }
  } else if (info[0].IsObject()) {
    // {"input0" : {buffer : inputbuffer0} }
    Napi::Object item = info[0].As<Napi::Object>();
    std::string name;
    WNNInput* input = ParseOperand<WNNInput>(item, name);
    inputs_.push_back(input);
    wnnInputsSetInput(inputs, name.data(), input);
  }

  WNNOutputs outputs = nullptr;
  if (info.Length() > 1) {
    outputs = dawn_native::CreateOutputs();
    if (info[1].IsArray()) {
      Napi::Array output_array = info[1].As<Napi::Array>();
      // [{"output0" : {buffer : outputbuffer0} }]
      for (size_t i = 0; i < output_array.Length(); ++i) {
        Napi::Object item = output_array.Get(i).As<Napi::Object>();
        std::string name;
        WNNOutput* output = ParseOperand<WNNOutput>(item, name);
        outputs_.push_back(output);
        wnnOutputsSetOutput(outputs, name.data(), output);
      }
    } else if (info[1].IsObject()) {
      // {"output0" : {buffer : outputbuffer0} }
      Napi::Object item = info[1].As<Napi::Object>();
      std::string name;
      WNNOutput* output = ParseOperand<WNNOutput>(item, name);
      outputs_.push_back(output);
      wnnOutputsSetOutput(outputs, name.data(), output);
    }
  }

  outputs = wnnCompilationCompute(compilation_, inputs,
      [](WNNOutputs outputs, void* userData){
      }, reinterpret_cast<void*>(this), outputs);
  
  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  if (info.Length() == 1) {
    size_t number = wnnOutputsGetOutputsNumber(outputs);
    if (number == 1) {
      deferred.Resolve(OutputItem(env, outputs, 0));
    } else {
      Napi::Array output_array = Napi::Array::New(env);
      for (size_t i = 0; i < number; ++i) {
        output_array[i] = OutputItem(env, outputs, i);
      }
      deferred.Resolve(output_array);
    }
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
