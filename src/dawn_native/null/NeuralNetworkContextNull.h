#ifndef WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_
#define WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_

#include "dawn_native/Compilation.h"
#include "dawn_native/Model.h"
#include "dawn_native/ModelBuilder.h"
#include "dawn_native/NeuralNetworkContext.h"

namespace dawn_native { namespace null {

    // NeuralNetworkContext
    class NeuralNetworkContext : public NeuralNetworkContextBase {
      public:
        NeuralNetworkContext() = default;
        ~NeuralNetworkContext() override = default;

      private:
        ModelBuilderBase* CreateModelBuilderImpl() override;
    };

    // ModelBuilder
    class ModelBuilder : public ModelBuilderBase {
      public:
        explicit ModelBuilder(NeuralNetworkContextBase* context);
        ~ModelBuilder() override = default;

      private:
        ModelBase* CreateModelImpl() override;
    };

    // Model
    class Model : public ModelBase {
      public:
        explicit Model(ModelBuilder* model_builder);
        ~Model() override = default;

      private:
        void CompileImpl(WNNCompileCallback callback,
                         void* userdata,
                         CompilationOptions const* options) override;
    };

    // Compilation
    class Compilation : public CompilationBase {
      public:
        Compilation() = default;
        ~Compilation() override = default;

      private:
        void ComputeImpl(NamedInputsBase* inputs,
                         WNNComputeCallback callback,
                         void* userdata,
                         NamedOutputsBase* outputs = nullptr) override;
    };

}}  // namespace dawn_native::null

#endif  // WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_
