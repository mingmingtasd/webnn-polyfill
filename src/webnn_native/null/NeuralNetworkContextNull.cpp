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
        return new NeuralNetworkContext();
    }

    ModelBuilderBase* NeuralNetworkContext::CreateModelBuilderImpl() {
        return new ModelBuilder(this);
    }

    // ModelBuilder
    ModelBuilder::ModelBuilder(NeuralNetworkContextBase* context) : ModelBuilderBase(context) {
    }

    ModelBase* ModelBuilder::CreateModelImpl() {
        return new Model(this);
    }

    // Model
    Model::Model(ModelBuilder* model_builder) : ModelBase(model_builder) {
    }

    void Model::CompileImpl(WebnnCompileCallback callback,
                            void* userdata,
                            CompilationOptions const* options) {
    }

    // Compilation
    void Compilation::ComputeImpl(NamedInputsBase* inputs,
                                  WebnnComputeCallback callback,
                                  void* userdata,
                                  NamedOutputsBase* outputs) {
    }

}}  // namespace webnn_native::null
