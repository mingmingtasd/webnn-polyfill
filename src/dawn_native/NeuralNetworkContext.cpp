
#include "dawn_native/NeuralNetworkContext.h"

#include <sstream>

#include "dawn_native/ValidationUtils_autogen.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

    NeuralNetworkContextBase::NeuralNetworkContextBase() {
        root_error_scope_ = AcquireRef(new ErrorScope());
        current_error_scope_ = root_error_scope_.Get();
    }

    ModelBuilderBase* NeuralNetworkContextBase::CreateModelBuilder() {
        return CreateModelBuilderImpl();
    }

    ModelBuilderBase* NeuralNetworkContextBase::CreateModelBuilderImpl() {
        UNREACHABLE();
    }

    void NeuralNetworkContextBase::PushErrorScope(wnn::ErrorFilter filter) {
        if (ConsumedError(ValidateErrorFilter(filter))) {
            return;
        }
        current_error_scope_ = AcquireRef(new ErrorScope(filter, current_error_scope_.Get()));
    }

    bool NeuralNetworkContextBase::PopErrorScope(wnn::ErrorCallback callback, void* userdata) {
        if (DAWN_UNLIKELY(current_error_scope_.Get() == root_error_scope_.Get())) {
            return false;
        }
        current_error_scope_->SetCallback(callback, userdata);
        current_error_scope_ = Ref<ErrorScope>(current_error_scope_->GetParent());

        return true;
    }

    void NeuralNetworkContextBase::SetUncapturedErrorCallback(wnn::ErrorCallback callback,
                                                              void* userdata) {
        root_error_scope_->SetCallback(callback, userdata);
    }

    void NeuralNetworkContextBase::HandleError(std::unique_ptr<ErrorData> error) {
        ASSERT(error != nullptr);
        std::ostringstream ss;
        ss << error->GetMessage();
        for (const auto& callsite : error->GetBacktrace()) {
            ss << "\n    at " << callsite.function << " (" << callsite.file << ":" << callsite.line
               << ")";
        }

        // Still forward device loss and internal errors to the error scopes so they
        // all reject.
        current_error_scope_->HandleError(ToWNNErrorType(error->GetType()), ss.str().c_str());
    }

}  // namespace dawn_native
