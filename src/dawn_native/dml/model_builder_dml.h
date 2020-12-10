#ifndef WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_
#define WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_

#include "dawn_native/ModelBuilder.h"

namespace pydml { class Device; }

namespace dawn_native {

namespace dml {

class ModelBuilder : public ModelBuilderBase {
public:
  ~ModelBuilder() override = default;
  ModelBuilder() = default;

private:
  ModelBase *CreateModelImpl(NamedOperandsBase const *named_operand) override;

  static pydml::Device* g_dml_device_;
};

} // namespace dml

} // namespace dawn_native

#endif // WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_
