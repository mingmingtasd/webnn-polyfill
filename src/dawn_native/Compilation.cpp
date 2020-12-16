
#include "dawn_native/Operand.h"

#include "common/Assert.h"
#include "common/Log.h"
#include "dawn_native/NamedResults.h"

namespace dawn_native {

void CompilationBase::Compute(NamedInputsBase *inputs,
                              WNNComputeCallback callback,
                              void *userdata, NamedOutputsBase *outputs) {
  WNNComputeStatus status;
  ResultOrError<Ref<NamedResultsBase>> result_or_error =
      ComputeImpl(inputs, outputs, &status);
  if (result_or_error.IsError()) {
    std::unique_ptr<ErrorData> error = result_or_error.AcquireError();
    callback(status, nullptr, error->GetMessage().c_str(), userdata);
  } else {
    Ref<NamedResultsBase> results = result_or_error.AcquireSuccess();
    callback(status, reinterpret_cast<WNNNamedResults>(results.Detach()),
             nullptr, userdata);
  }
}

} // namespace dawn_native
