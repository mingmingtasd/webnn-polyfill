#include "dawn_native/null/neural_network_context_null.h"
#include "common/RefCounted.h"

namespace dawn_native {

namespace null {

// NeuralNetworkContext
NeuralNetworkContextBase *Create() {
  Ref<NeuralNetworkContextBase> context =
      AcquireRef(new NeuralNetworkContext());
  return context.Detach();
}

ModelBuilderBase *NeuralNetworkContext::CreateModelBuilderImpl() {
  Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder(this));
  return builder.Detach();
}

// ModelBuilder
ModelBuilder::ModelBuilder(NeuralNetworkContextBase *context)
    : ModelBuilderBase(context) {}

ModelBase *ModelBuilder::CreateModelImpl() {
  Ref<ModelBase> model = AcquireRef(new Model(this));
  return model.Detach();
}

// Model
Model::Model(ModelBuilder *model_builder) : ModelBase(model_builder) {
}

void Model::CompileImpl(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
}

// Compilation
void Compilation::ComputeImpl(
    NamedInputsBase *inputs, WNNComputeCallback callback,
    void *userdata,
    NamedOutputsBase *outputs) {
}

} // namespace null

} // namespace dawn_native
