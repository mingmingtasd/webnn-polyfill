
#include "dawn_native/dml/neural_network_context_dml.h"

#include "common/RefCounted.h"
#include "dawn_native/dml/model_builder_dml.h"

namespace dawn_native {

namespace dml {

NeuralNetworkContextBase *Create() {
  Ref<NeuralNetworkContextBase> context =
      AcquireRef(new NeuralNetworkContext());
  return context.Detach();
}

ModelBuilderBase *NeuralNetworkContext::CreateModelBuilderImpl() {
  Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder(this));
  return builder.Detach();
}

} // namespace dml

} // namespace dawn_native
