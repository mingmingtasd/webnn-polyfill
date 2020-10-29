
#include "dawn_native/ie/compilation_ie.h"

#include <vector>

#include "common/Log.h"
#include "dawn_native/Inputs.h"
#include "dawn_native/Operand.h"
#include "dawn_native/Outputs.h"
#include "ienn_symbol_table.h"

namespace dawn_native {

namespace ie {

Compilation::Compilation(Ref<Model> model) : model_(model) {
  // Create compilation for IE backend.
  IEStatusCode code = IE(ie_create_compilation)(
      model->GetInferenceEngineModel(), &ie_compilation_);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to create compilation for IE.";
    return;
  }
}

Compilation::~Compilation() {
  // IE(ie_compilation_free)(ie_compilation_);
}

void Compilation::ComputeImpl(InputsBase *inputs, WNNComputeCallback callback,
                              OutputsBase *outputs) {
  // Set input data to nGraph.
  for (auto &input : inputs->GetInputs()) {
    OperandBase *operand = model_->GetNamedOperand(input.first);
    ie_operand_t ie_operand;
    ie_operand.name = const_cast<char *>(operand->GetName().c_str());
    IEStatusCode code = IE(ie_compilation_set_input)(
        ie_compilation_, &ie_operand, input.second->buffer, input.second->size);
    if (code != IEStatusCode::OK) {
      dawn::ErrorLog() << "Failing to set input for IE.";
      return;
    }
  }

  // Compute the compiled model.
  IEStatusCode code = IE(ie_compilation_compute)(ie_compilation_);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to compute for IE.";
    return;
  }

  // Get Data from nGraph with output.
  // TODO(junwei). new memory for output data.
  if (outputs == nullptr) {
    outputs = new OutputsBase();
  }
  for (auto &output : outputs->GetOutputs()) {
    OperandBase *operand = model_->GetNamedOperand(output.first);
    ie_operand_t ie_operand;
    ie_operand.name = const_cast<char *>(operand->GetName().c_str());
    IEStatusCode code = IE(ie_compilation_get_output)(
        ie_compilation_, &ie_operand, output.second->buffer,
        output.second->size);
    if (code != IEStatusCode::OK) {
      dawn::ErrorLog() << "Failing to get output for IE.";
      return;
    }
  }
  callback(reinterpret_cast<WNNOutputs>(outputs));
}

} // namespace ie

} // namespace dawn_native
