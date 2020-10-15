
#include "dawn_native/ie/model_builder_ie.h"

#include "common/Log.h"
#include "dawn_native/ie/model_ie.h"

namespace dawn_native {

namespace ie {

ModelBuilderBase *Create() {
  Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder());
  return builder.Detach();
}

ModelBase *ModelBuilder::CreateModelImpl(NamedOperand const *named_operand,
                                         size_t size) {
  Ref<ModelBase> model = AcquireRef(new Model(named_operand, size));
  return model.Detach();
}

} // namespace ie

} // namespace dawn_native
