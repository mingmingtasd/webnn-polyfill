#include "compilation.h"

#include <iostream>

#include "DescriptorDecoder.h"
#include "model.h"

// Hold Promise::Deferred with AsyncWorker.
class ComputeAsyncWorker : public Napi::AsyncWorker {
 public:
  ComputeAsyncWorker(Napi::Env& env,
                     Napi::Promise::Deferred& deferred,
                     WNNCompilation compilation,
                     WNNNamedInputs named_inputs,
                     WNNNamedOutputs named_outputs,
                     std::vector<std::string>& output_names,
                     const Napi::CallbackInfo& info)
      : Napi::AsyncWorker(env),
        env_(env),
        deferred_(deferred),
        compilation_(compilation),
        named_inputs_(named_inputs),
        named_outputs_(named_outputs),
        output_names_(output_names) {
    if (info.Length() > 1) {
      output_objects_ = info[1].ToObject();
    }
  }
  ~ComputeAsyncWorker() { wnnNamedResultsRelease(named_results_); }

  void Execute() {
    wnnCompilationCompute(
        compilation_, named_inputs_,
        [](WNNComputeStatus status, WNNNamedResults results,
           char const* message, void* user_data) {
          ComputeAsyncWorker* compute_worker =
              reinterpret_cast<ComputeAsyncWorker*>(user_data);
          compute_worker->SetNamedResults(results);
        },
        reinterpret_cast<void*>(this), named_outputs_);
  }
  void OnOK() {
    if (output_objects_.IsEmpty()) {
      Napi::Object obj = Napi::Object::New(env_);
      for (auto& name : output_names_) {
        WNNResult result = wnnNamedResultsGet(named_results_, name.data());
        obj.Set(name, OutputItem(env_, result));
        wnnResultRelease(result);
      }
      deferred_.Resolve(obj);
    } else {
      deferred_.Resolve(output_objects_);
    }
  }
  void SetNamedResults(WNNNamedResults named_results) {
    named_results_ = named_results;
  }

 private:
  Napi::Object OutputItem(const Napi::Env& env, WNNResult result) {
    Napi::Object item = Napi::Object::New(env);
    size_t buffer_size = wnnResultBufferSize(result);
    Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(
        env, const_cast<void*>(wnnResultBuffer(result)), buffer_size);
    Napi::Float32Array output_buffer =
        Napi::Float32Array::New(env, buffer_size / sizeof(float), buffer, 0);
    item.Set("buffer", output_buffer);
    return item;
  }
  Napi::Env env_;
  Napi::Promise::Deferred deferred_;
  WNNCompilation compilation_;
  WNNNamedInputs named_inputs_;
  WNNNamedOutputs named_outputs_;
  std::vector<std::string>& output_names_;
  WNNNamedResults named_results_;
  Napi::Object output_objects_;
};

Napi::FunctionReference Compilation::constructor;

Compilation::Compilation(const Napi::CallbackInfo& info)
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

Napi::Value Compilation::Compute(const Napi::CallbackInfo& info) {
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

  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  Model* model = Napi::ObjectWrap<Model>::Unwrap(model_object_.Value());
  compute_worker_ =
      new ComputeAsyncWorker(env, deferred, compilation_, named_inputs,
                             named_outputs, model->GetOutputName(), info);
  compute_worker_->Queue();

  return deferred.Promise();
}

Napi::Object Compilation::Initialize(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  Napi::Function func = DefineClass(
      env, "Compilation",
      {InstanceMethod("compute", &Compilation::Compute, napi_enumerable)});
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Compilation", func);
  return exports;
}
