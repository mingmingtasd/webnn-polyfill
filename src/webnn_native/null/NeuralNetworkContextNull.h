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

#ifndef WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_
#define WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_

#include "webnn_native/Compilation.h"
#include "webnn_native/Model.h"
#include "webnn_native/ModelBuilder.h"
#include "webnn_native/NeuralNetworkContext.h"

namespace webnn_native { namespace null {

    // NeuralNetworkContext
    class NeuralNetworkContext : public NeuralNetworkContextBase {
      public:
        NeuralNetworkContext() = default;
        ~NeuralNetworkContext() override = default;

      private:
        ModelBuilderBase* CreateModelBuilderImpl() override;
    };

    // ModelBuilder
    class ModelBuilder : public ModelBuilderBase {
      public:
        explicit ModelBuilder(NeuralNetworkContextBase* context);
        ~ModelBuilder() override = default;

      private:
        ModelBase* CreateModelImpl() override;
    };

    // Model
    class Model : public ModelBase {
      public:
        explicit Model(ModelBuilder* model_builder);
        ~Model() override = default;

      private:
        void CompileImpl(WebnnCompileCallback callback,
                         void* userdata,
                         CompilationOptions const* options) override;
    };

    // Compilation
    class Compilation : public CompilationBase {
      public:
        Compilation() = default;
        ~Compilation() override = default;

      private:
        void ComputeImpl(NamedInputsBase* inputs,
                         WebnnComputeCallback callback,
                         void* userdata,
                         NamedOutputsBase* outputs = nullptr) override;
    };

}}  // namespace webnn_native::null

#endif  // WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_