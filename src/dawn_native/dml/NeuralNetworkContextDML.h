#ifndef WEBNN_NATIVE_DML_NEURAL_NETWORK_CONTEXT_DML_H_
#define WEBNN_NATIVE_DML_NEURAL_NETWORK_CONTEXT_DML_H_

#include "dawn_native/NeuralNetworkContext.h"

namespace dawn_native { namespace dml {

    class NeuralNetworkContext : public NeuralNetworkContextBase {
      public:
        NeuralNetworkContext() = default;
        ~NeuralNetworkContext() override = default;

        ModelBuilderBase* CreateModelBuilderImpl() override;

      private:
    };

}}  // namespace dawn_native::dml

#endif  // WEBNN_NATIVE_DML_NEURAL_NETWORK_CONTEXT_DML_H_
