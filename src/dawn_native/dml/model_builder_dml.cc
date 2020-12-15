#include "dawn_native/dml/model_builder_dml.h"

#include "common/Log.h"
#include "dawn_native/dml/model_dml.h"

#include "dawn_native/dml/deps/src/precomp.h"

namespace dawn_native {

namespace dml {

ModelBuilderBase *Create() {
  Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder());
  return builder.Detach();
}

ModelBase *ModelBuilder::CreateModelImpl() {
  Ref<ModelBase> model = AcquireRef(new Model());
  return model.Detach();
}

} // namespace dml

} // namespace dawn_native
