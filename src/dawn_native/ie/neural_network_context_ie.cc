
#include "dawn_native/ie/neural_network_context_ie.h"

#include "common/RefCounted.h"
#include "dawn_native/ie/model_builder_ie.h"

namespace dawn_native {

namespace ie {

NeuralNetworkContextBase *Create() {
  Ref<NeuralNetworkContextBase> context =
      AcquireRef(new NeuralNetworkContext());
  return context.Detach();
}

ModelBuilderBase *NeuralNetworkContext::CreateModelBuilderImpl() {
  Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder(this));
  return builder.Detach();
}

} // namespace ie

} // namespace dawn_native
