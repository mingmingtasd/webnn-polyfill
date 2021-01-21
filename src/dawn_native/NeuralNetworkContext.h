#ifndef WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_H_
#define WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_H_

#include "common/RefCounted.h"
#include "dawn_native/Error.h"
#include "dawn_native/ErrorScope.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

    class NeuralNetworkContextBase : public RefCounted {
      public:
        NeuralNetworkContextBase();
        virtual ~NeuralNetworkContextBase() = default;

        bool ConsumedError(MaybeError maybeError) {
            if (DAWN_UNLIKELY(maybeError.IsError())) {
                HandleError(maybeError.AcquireError());
                return true;
            }
            return false;
        }

        // Dawn API
        ModelBuilderBase* CreateModelBuilder();
        void PushErrorScope(wnn::ErrorFilter filter);
        bool PopErrorScope(wnn::ErrorCallback callback, void* userdata);
        void SetUncapturedErrorCallback(wnn::ErrorCallback callback, void* userdata);

      private:
        void HandleError(std::unique_ptr<ErrorData> error);
        virtual ModelBuilderBase* CreateModelBuilderImpl();

        Ref<ErrorScope> root_error_scope_;
        Ref<ErrorScope> current_error_scope_;
    };

}  // namespace dawn_native

#endif  // WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_H_
