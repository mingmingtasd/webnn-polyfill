
#include "dawn_native/NeuralNetworkContext.h"

#include <sstream>

namespace dawn_native {

NeuralNetworkContextBase::NeuralNetworkContextBase() {
  root_error_scope_ = AcquireRef(new ErrorScope());
}

ModelBuilderBase *NeuralNetworkContextBase::CreateModelBuilder() {
  return CreateModelBuilderImpl();
}

ModelBuilderBase *NeuralNetworkContextBase::CreateModelBuilderImpl() {
  UNREACHABLE();
}

void NeuralNetworkContextBase::SetUncapturedErrorCallback(
    wnn::ErrorCallback callback, void *userdata) {
  root_error_scope_->SetCallback(callback, userdata);
}

void NeuralNetworkContextBase::HandleError(std::unique_ptr<ErrorData> error) {
  ASSERT(error != nullptr);
  std::ostringstream ss;
  ss << error->GetMessage();
  for (const auto &callsite : error->GetBacktrace()) {
    ss << "\n    at " << callsite.function << " (" << callsite.file << ":"
       << callsite.line << ")";
  }

  // Still forward device loss and internal errors to the error scopes so they
  // all reject.
  root_error_scope_->HandleError(ToWNNErrorType(error->GetType()),
                                 ss.str().c_str());
}

} // namespace dawn_native
