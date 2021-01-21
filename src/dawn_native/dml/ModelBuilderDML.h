#ifndef WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_
#define WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_

#include "dawn_native/ModelBuilder.h"

namespace dawn_native { namespace dml {

    class ModelBuilder : public ModelBuilderBase {
      public:
        explicit ModelBuilder(NeuralNetworkContextBase* context);
        ~ModelBuilder() override = default;

      private:
        ModelBase* CreateModelImpl() override;
    };

}}  // namespace dawn_native::dml

#endif  // WEBNN_NATIVE_DML_MODEL_BUILDER_DML_H_
