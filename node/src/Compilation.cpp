#include "Compilation.h"

#include <iostream>

#include "Model.h"

// Hold Promise::Deferred with AsyncWorker.
class ComputeAsyncWorker : public Napi::AsyncWorker {
 public:
   ComputeAsyncWorker(Napi::Env& env,
                      Napi::Promise::Deferred& deferred,
                      WebnnCompilation compilation,
                      std::vector<WebnnInput> inputs,
                      std::vector<WebnnOutput> outputs,
                      WebnnNamedInputs named_inputs,
                      WebnnNamedOutputs named_outputs,
                      std::vector<std::string>& output_names)
       : Napi::AsyncWorker(env),
         env_(env),
         deferred_(deferred),
         compilation_(compilation),
         inputs_(std::move(inputs)),
         outputs_(std::move(outputs)),
         named_inputs_(named_inputs),
         named_outputs_(named_outputs),
         output_names_(output_names) {
   }

  ~ComputeAsyncWorker() { webnnNamedResultsRelease(named_results_); }

  void Execute() {
    webnnCompilationCompute(
        compilation_, named_inputs_,
        [](WebnnComputeStatus status, WebnnNamedResults results,
           char const* message, void* user_data) {
          ComputeAsyncWorker* compute_worker =
              reinterpret_cast<ComputeAsyncWorker*>(user_data);
          compute_worker->SetNamedResults(results);
        },
        reinterpret_cast<void*>(this), named_outputs_);
  }
  void OnOK() {
      if (outputs_.empty()) {
          Napi::Object obj = Napi::Object::New(env_);
          for (auto& name : output_names_) {
              WebnnResult result = webnnNamedResultsGet(named_results_, name.data());
              obj.Set(name, OutputItem(env_, result));
              webnnResultRelease(result);
          }
          deferred_.Resolve(obj);
      } else {
          deferred_.Resolve(Env().Null());
      }
  }
  void SetNamedResults(WebnnNamedResults named_results) {
    named_results_ = named_results;
  }

 private:
  Napi::Object OutputItem(const Napi::Env& env, WebnnResult result) {
    Napi::Object item = Napi::Object::New(env);
    size_t buffer_size = webnnResultBufferSize(result);
    Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(
        env, const_cast<void*>(webnnResultBuffer(result)), buffer_size);
    Napi::Float32Array output_buffer =
        Napi::Float32Array::New(env, buffer_size / sizeof(float), buffer, 0);
    item.Set("buffer", output_buffer);
    return item;
  }
  Napi::Env env_;
  Napi::Promise::Deferred deferred_;
  WebnnCompilation compilation_;
  std::vector<WebnnInput> inputs_;
  std::vector<WebnnOutput> outputs_;
  WebnnNamedInputs named_inputs_;
  WebnnNamedOutputs named_outputs_;
  std::vector<std::string>& output_names_;
  WebnnNamedResults named_results_;
};

Napi::FunctionReference Compilation::constructor;

Compilation::Compilation(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Compilation>(info) {
  this->model_object_.Reset(info[0].ToObject(), 1);
}

Compilation::~Compilation() {
  this->model_object_.Reset();
  webnnCompilationRelease(compilation_);
}

void Compilation::SetCompilation(WebnnCompilation Compilation) {
  compilation_ = Compilation;
}

WebnnCompilation Compilation::GetCompilation() {
  return compilation_;
}

Napi::Value Compilation::Compute(const Napi::CallbackInfo& info) {
  WebnnNamedInputs named_inputs = webnn_native::CreateNamedInputs();
  // The WebnnInput struct need to be kept until compute.
  std::vector<WebnnInput> inputs;
  if (info[0].IsObject()) {
      Napi::Object obj = info[0].As<Napi::Object>();
      Napi::Array property_names = obj.GetPropertyNames();

      for (size_t j = 0; j < property_names.Length(); ++j) {
        std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
        inputs.push_back(ParseOperand<WebnnInput>(obj, name));
        webnnNamedInputsSet(named_inputs, name.data(), &inputs[j]);
      }
  }

  // The WebnnOutput struct need to be kept until compute.
  std::vector<WebnnOutput> outputs;
  WebnnNamedOutputs named_outputs = nullptr;
  if (info.Length() > 1 && info[1].IsObject()) {
      named_outputs = webnn_native::CreateNamedOutputs();
      Napi::Object obj = info[1].As<Napi::Object>();
      Napi::Array property_names = obj.GetPropertyNames();
      for (size_t j = 0; j < property_names.Length(); ++j) {
          std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
          outputs.push_back(ParseOperand<WebnnOutput>(obj, name));
          webnnNamedOutputsSet(named_outputs, name.data(), &outputs[j]);
      }
  }
  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  Model* model = Napi::ObjectWrap<Model>::Unwrap(model_object_.Value());
  compute_worker_ =
      new ComputeAsyncWorker(env, deferred, compilation_, std::move(inputs), std::move(outputs),
                             named_inputs, named_outputs, model->GetOutputName());
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
