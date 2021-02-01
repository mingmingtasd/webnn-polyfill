// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "webnn_native/null/NeuralNetworkContextNull.h"
#include "common/RefCounted.h"

namespace webnn_native { namespace null {

    // NeuralNetworkContext
    NeuralNetworkContextBase* Create() {
        Ref<NeuralNetworkContextBase> context = AcquireRef(new NeuralNetworkContext());
        return context.Detach();
    }

    ModelBuilderBase* NeuralNetworkContext::CreateModelBuilderImpl() {
        Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder(this));
        return builder.Detach();
    }

    // ModelBuilder
    ModelBuilder::ModelBuilder(NeuralNetworkContextBase* context) : ModelBuilderBase(context) {
    }

    ModelBase* ModelBuilder::CreateModelImpl() {
        Ref<ModelBase> model = AcquireRef(new Model(this));
        return model.Detach();
    }

    // Model
    Model::Model(ModelBuilder* model_builder) : ModelBase(model_builder) {
    }

    void Model::CompileImpl(WNNCompileCallback callback,
                            void* userdata,
                            CompilationOptions const* options) {
    }

    // Compilation
    void Compilation::ComputeImpl(NamedInputsBase* inputs,
                                  WNNComputeCallback callback,
                                  void* userdata,
                                  NamedOutputsBase* outputs) {
    }

}}  // namespace webnn_native::null
