#include "Compilation.h"

#include <iostream>
#include <map>

#include "Model.h"

// Hold Promise::Deferred with AsyncWorker.
class ComputeAsyncWorker : public Napi::AsyncWorker {
 public:
   ComputeAsyncWorker(Napi::Env& env,
                      Napi::Promise::Deferred& deferred,
                      WebnnCompilation compilation,
                      std::map<std::string, WebnnInput> inputs,
                      std::map<std::string, WebnnOutput> outputs,
                      std::vector<std::string>& output_names)
       : Napi::AsyncWorker(env),
         env_(env),
         deferred_(deferred),
         compilation_(compilation),
         inputs_(std::move(inputs)),
         outputs_(std::move(outputs)),
         output_names_(output_names) {
   }

  ~ComputeAsyncWorker() { webnnNamedResultsRelease(named_results_); }

  void Execute() {
      WebnnNamedInputs named_inputs = webnn_native::CreateNamedInputs();
      for (auto& input : inputs_) {
          webnnNamedInputsSet(named_inputs, input.first.data(), &input.second);
      }
      WebnnNamedOutputs named_outputs =
          outputs_.empty() ? nullptr : webnn_native::CreateNamedOutputs();
      for (auto& output : outputs_) {
          webnnNamedOutputsSet(named_outputs, output.first.data(), &output.second);
      }

      webnnCompilationCompute(
          compilation_, named_inputs,
          [](WebnnComputeStatus status, WebnnNamedResults results, char const* message,
             void* user_data) {
              ComputeAsyncWorker* compute_worker = reinterpret_cast<ComputeAsyncWorker*>(user_data);
              compute_worker->SetNamedResults(results);
          },
          reinterpret_cast<void*>(this), named_outputs);
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

    size_t dimensions_size = webnnResultDimensionsSize(result);
    Napi::Array dimensions = Napi::Array::New(env, dimensions_size);
    for (size_t i = 0; i < dimensions_size; ++i) {
        dimensions[i] = Napi::Number::New(env, webnnResultDimensions(result)[i]);
    }
    item.Set("dimensions", dimensions);
    return item;
  }
  Napi::Env env_;
  Napi::Promise::Deferred deferred_;
  WebnnCompilation compilation_;
  std::map<std::string, WebnnInput> inputs_;
  std::map<std::string, WebnnOutput> outputs_;
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
  // The WebnnInput struct need to be kept until compute.
  std::map<std::string, WebnnInput> inputs;
  if (info[0].IsObject()) {
      Napi::Object obj = info[0].As<Napi::Object>();
      inputs = GetNamedOperands<WebnnInput>(obj);
  }

  // The WebnnOutput struct need to be kept until compute.
  std::map<std::string, WebnnOutput> outputs;
  if (info.Length() > 1 && info[1].IsObject()) {
      Napi::Object obj = info[1].As<Napi::Object>();
      outputs = GetNamedOperands<WebnnOutput>(obj);
  }
  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  Model* model = Napi::ObjectWrap<Model>::Unwrap(model_object_.Value());
  compute_worker_ = new ComputeAsyncWorker(env, deferred, compilation_, std::move(inputs),
                                           std::move(outputs), model->GetOutputName());
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
