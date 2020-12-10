#include "dawn_native/dml/model_builder_dml.h"

#include "common/Log.h"
#include "dawn_native/dml/model_dml.h"

#include "dawn_native/dml/deps/src/precomp.h"

namespace dawn_native {

namespace dml {

pydml::Device* ModelBuilder::g_dml_device_ = nullptr;

ModelBuilderBase *Create() {
  Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder());
  return builder.Detach();
}

ModelBase *ModelBuilder::CreateModelImpl(
    NamedOperandsBase const *named_operands) {
  if (g_dml_device_ == nullptr) {
    g_dml_device_ = new pydml::Device();
  }
  Ref<ModelBase> model = AcquireRef(new Model(named_operands));
  return model.Detach();
}

} // namespace dml

} // namespace dawn_native
