#ifndef WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_
#define WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_

#include "dawn_native/ModelBuilder.h"

namespace dawn_native {

namespace dml {

class ModelBuilder : public ModelBuilderBase {
public:
  ~ModelBuilder() override = default;
  ModelBuilder() = default;

private:
  ModelBase *CreateModelImpl() override;
};

} // namespace dml

} // namespace dawn_native

#endif // WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_
