#ifndef WEBNN_NATIVE_IE_MODEL_BUILDER_IE_H_
#define WEBNN_NATIVE_IE_MODEL_BUILDER_IE_H_

#include "dawn_native/ModelBuilder.h"

namespace dawn_native {

namespace ie {

class ModelBuilder : public ModelBuilderBase {
public:
  explicit ModelBuilder(NeuralNetworkContextBase *context);
  ~ModelBuilder() override = default;

private:
  ModelBase *CreateModelImpl() override;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_MODEL_BUILDER_IE_H_