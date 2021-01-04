#include "dawn_native/dml/compilation_dml.h"

#include <vector>

#include "common/Log.h"
#include "dawn_native/Operand.h"
#include "dawn_native/Result.h"
#include "dawn_native/NamedResults.h"
#include "dawn_native/dml/deps/src/precomp.h"

namespace dawn_native {

namespace dml {

class Result : public ResultBase {
public:
  using ResultBase::Reference;
  ~Result() {
    free(buffer_);
  }
};

Compilation::Compilation(const Ref<Model>& model) : model_(model) {
  std::vector<::dml::Expression> outputs;
  for (auto& output : model_->outputs_) {
    outputs.push_back(output.second);
  }
  // TODO(nhu): investigate other execution flag,
  // e.g. DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
  compiled_model_.reset(
      new pydml::CompiledModel(
          *(model_->graph_), DML_EXECUTION_FLAG_NONE, outputs));
}

void Compilation::ComputeImpl(
    NamedInputsBase *inputs, WNNComputeCallback callback,
    void *userdata,
    NamedOutputsBase *outputs) {
  // FIXME(nhu): implement async
  for (auto &input : inputs->GetRecords()) {
    ::pydml::Binding* input_binding = model_->inputs_.at(input.first);
    input_binding->data.buffer_ = const_cast<void*>(input.second->buffer);
    input_binding->data.size_ = input.second->size;
  }
  std::vector<pydml::Binding*> input_bindings;
  for (auto &binding : model_->bindings_) {
    input_bindings.push_back(binding.get());
  }
  std::vector<::dml::Expression*> output_expressions;
  std::vector<std::string> output_names;
  if (outputs != nullptr) {
    for (auto& output : outputs->GetRecords()) {
      output_names.push_back(output.first);
      output_expressions.push_back(&(model_->outputs_.at(output.first)));
    }
  } else {
    for (auto& output : model_->outputs_) {
      output_names.push_back(output.first);
      output_expressions.push_back(&(output.second));
    }
  }
  std::vector<pydml::TensorData*> output_tensors =
      model_->device_->Compute(
          compiled_model_->op.Get(),
          input_bindings, output_expressions);

  Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
  for (size_t i = 0; i < output_names.size(); ++i) {
    std::string output_name = output_names[i];
    pydml::TensorData* tensor = output_tensors[i];
    void *output_buffer = tensor->Get();
    size_t buffer_length = tensor->Size();
    std::vector<int32_t> dimensions = output_tensors[i]->dimensions_;
    Ref<ResultBase> result = AcquireRef(
        new Result::ResultBase(output_buffer, buffer_length, dimensions));
    results->Set(output_name.c_str(), result.Detach());
    if (outputs != nullptr) {
      const Output* output = outputs->GetRecords().at(output_name);
      if (output->size >= buffer_length) {
        memcpy(output->buffer, output_buffer, buffer_length);
      }
    }
    delete tensor;
  }
  WNNComputeStatus status = WNNComputeStatus_Success;
  callback(status, reinterpret_cast<WNNNamedResults>(results.Detach()),
           nullptr, userdata);
  return;
}

}  // namespace dml
}  // namespace dawn_native
