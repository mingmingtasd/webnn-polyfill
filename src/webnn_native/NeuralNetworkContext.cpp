// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "webnn_native/NeuralNetworkContext.h"

#include <sstream>

#include "webnn_native/ValidationUtils_autogen.h"
#include "webnn_native/webnn_platform.h"

namespace webnn_native {

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

}  // namespace webnn_native
