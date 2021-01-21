#include "dawn_native/dml/ModelBuilderDML.h"

#include "common/Log.h"
#include "dawn_native/dml/ModelDML.h"
#include "dawn_native/dml/deps/src/precomp.h"

namespace dawn_native { namespace dml {

    ModelBuilder::ModelBuilder(NeuralNetworkContextBase* context) : ModelBuilderBase(context) {
    }

    ModelBase* ModelBuilder::CreateModelImpl() {
        Ref<ModelBase> model = AcquireRef(new Model(this));
        return model.Detach();
    }

}}  // namespace dawn_native::dml
