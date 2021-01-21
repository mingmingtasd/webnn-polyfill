
#include "dawn_native/dml/NeuralNetworkContextDML.h"

#include "common/RefCounted.h"
#include "dawn_native/dml/ModelBuilderDML.h"

namespace dawn_native { namespace dml {

    NeuralNetworkContextBase* Create() {
        Ref<NeuralNetworkContextBase> context = AcquireRef(new NeuralNetworkContext());
        return context.Detach();
    }

    ModelBuilderBase* NeuralNetworkContext::CreateModelBuilderImpl() {
        Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder(this));
        return builder.Detach();
    }

}}  // namespace dawn_native::dml
