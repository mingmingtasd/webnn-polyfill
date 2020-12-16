
#include "dawn_native/ie/model_builder_ie.h"

#include "common/Log.h"
#include "dawn_native/ie/model_ie.h"

namespace dawn_native {

namespace ie {

ModelBuilderBase *Create() {
  Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder());
  return builder.Detach();
}

ModelBase *ModelBuilder::CreateModelImpl() {
  Ref<ModelBase> model = AcquireRef(new Model());
  return model.Detach();
}

} // namespace ie

} // namespace dawn_native
