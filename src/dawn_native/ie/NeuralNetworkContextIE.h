#ifndef WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_IE_H_
#define WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_IE_H_

#include "dawn_native/NeuralNetworkContext.h"

namespace dawn_native { namespace ie {

    class NeuralNetworkContext : public NeuralNetworkContextBase {
      public:
        NeuralNetworkContext() = default;
        ~NeuralNetworkContext() override = default;

        ModelBuilderBase* CreateModelBuilderImpl() override;

      private:
    };

}}  // namespace dawn_native::ie

#endif  // WEBNN_NATIVE_NEURAL_NETWORK_CONTEXT_IE_H_
