
#include "dawn_native/ie/NeuralNetworkContextIE.h"

#include "common/RefCounted.h"
#include "dawn_native/ie/ModelBuilderIE.h"

namespace dawn_native { namespace ie {

    NeuralNetworkContextBase* Create() {
        Ref<NeuralNetworkContextBase> context = AcquireRef(new NeuralNetworkContext());
        return context.Detach();
    }

    ModelBuilderBase* NeuralNetworkContext::CreateModelBuilderImpl() {
        Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder(this));
        return builder.Detach();
    }

}}  // namespace dawn_native::ie
