#include "dawn_native/dml/model_builder_dml.h"

#include "common/Log.h"
#include "dawn_native/dml/model_dml.h"
#include "dawn_native/dml/deps/src/precomp.h"

namespace dawn_native {

namespace dml {

ModelBuilder::ModelBuilder(NeuralNetworkContextBase *context)
    : ModelBuilderBase(context) {}

ModelBase *ModelBuilder::CreateModelImpl() {
  Ref<ModelBase> model = AcquireRef(new Model(this));
  return model.Detach();
}

} // namespace dml

} // namespace dawn_native
