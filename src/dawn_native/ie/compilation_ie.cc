
#include "dawn_native/ie/compilation_ie.h"

#include <vector>

#include "common/Log.h"
#include "dawn_native/NamedResults.h"
#include "dawn_native/Operand.h"
#include "dawn_native/Result.h"
#include "error_ie.h"
#include "ienn_symbol_table.h"

namespace dawn_native {

namespace ie {

class Result : public ResultBase {
public:
  using ResultBase::Reference;
  ~Result() {
    ie_compilation_free_buffer(&buffer_);
  }
};

Compilation::Compilation(Ref<Model> model) : model_(model) {}

Compilation::~Compilation() { IE(ie_compilation_free)(ie_compilation_); }

MaybeError Compilation::Init(WNNCompileStatus *status) {
  *status = WNNCompileStatus_Error;
  // Create compilation for IE backend.
  IEStatusCode code = IE(ie_create_compilation)(
      model_->GetInferenceEngineModel(), &ie_compilation_);
  DAWN_TRY(CheckStatusCode(code, "IE create compilation"));
  *status = WNNCompileStatus_Success;
  return {};
}

ResultOrError<Ref<NamedResultsBase>>
Compilation::ComputeImpl(NamedInputsBase *inputs, NamedOutputsBase *outputs,
                         WNNComputeStatus *status) {
  *status = WNNComputeStatus_Error;
  // Set input data to nGraph.
  for (auto &input : inputs->GetRecords()) {
    ie_operand_t ie_operand;
    ie_operand.name =
        const_cast<char *>(model_->input_id_map_[input.first].c_str());
    IEStatusCode code = IE(ie_compilation_set_input)(
        ie_compilation_, &ie_operand, input.second->buffer, input.second->size);
    DAWN_TRY(CheckStatusCode(code, "IE set input to model"));
  }

  // Compute the compiled model.
  IEStatusCode code = IE(ie_compilation_compute)(ie_compilation_);
  DAWN_TRY(CheckStatusCode(code, "IE compute model"));

  // Get Data from nGraph with output.
  // TODO(junwei). new memory for output data.
  Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
  size_t output_number = model_->GetOutputsNumber();
  for (size_t i = 0; i < output_number; ++i) {
    std::string output_id = model_->GetOutputId(i);
    void *output_buffer;
    size_t buffer_length;
    IEStatusCode code = IE(ie_compilation_get_buffer)(
        ie_compilation_, output_id.data(), &output_buffer, &buffer_length);
    DAWN_TRY(CheckStatusCode(code, "IE get buffer"));
    // TODO(junwei): get the output dimensions;
    std::vector<int32_t> dimensions;
    Ref<ResultBase> result = AcquireRef(
        new Result::ResultBase(output_buffer, buffer_length, dimensions));
    std::string output_name = model_->output_name_map_[output_id];
    results->Set(output_name.c_str(), result.Detach());
    if (outputs != nullptr) {
      const Output *output = outputs->GetRecords().at(output_name);
      ie_operand_t ie_operand;
      ie_operand.name = const_cast<char *>(output_id.c_str());
      IEStatusCode code = IE(ie_compilation_get_output)(
          ie_compilation_, &ie_operand, output->buffer, output->size);
      DAWN_TRY(CheckStatusCode(code, "IE get output"));
    }
  }
  *status = WNNComputeStatus_Success;
  return std::move(results);
}

} // namespace ie

} // namespace dawn_native
