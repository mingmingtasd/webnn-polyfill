
#include "dawn_native/ie/ModelBuilderIE.h"

#include "common/Log.h"
#include "dawn_native/ie/ModelIE.h"

namespace dawn_native {

namespace ie {

ModelBuilder::ModelBuilder(NeuralNetworkContextBase *context)
    : ModelBuilderBase(context) {}

ModelBase *ModelBuilder::CreateModelImpl() {
  Ref<ModelBase> model = AcquireRef(new Model(this));
  return model.Detach();
}

} // namespace ie

} // namespace dawn_native
