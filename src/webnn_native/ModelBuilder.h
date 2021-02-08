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

#ifndef WEBNN_NATIVE_MODEL_BUILDER_H_
#define WEBNN_NATIVE_MODEL_BUILDER_H_

#include "common/RefCounted.h"
#include "webnn_native/Forward.h"
#include "webnn_native/NamedOperands.h"
#include "webnn_native/ObjectBase.h"
#include "webnn_native/webnn_platform.h"

#include <vector>

namespace webnn_native {

    class ModelBuilderBase : public ObjectBase {
      public:
        ModelBuilderBase(NeuralNetworkContextBase* context);
        virtual ~ModelBuilderBase() = default;

        // WebNN API
        OperandBase* Constant(OperandDescriptor const* desc, void const* value, size_t size);
        OperandBase* Input(char const* name, OperandDescriptor const* desc);
        OperandBase* Matmul(OperandBase* a, OperandBase* b);
        OperandBase* Add(OperandBase*, OperandBase*);
        OperandBase* Mul(OperandBase*, OperandBase*);
        OperandBase* Conv2d(OperandBase*, OperandBase*, Conv2dOptions const* options);
        OperandBase* AveragePool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* MaxPool2d(OperandBase*, Pool2dOptions const* options);
        OperandBase* Relu(OperandBase*);
        OperandBase* Reshape(OperandBase*, int32_t const*, size_t);
        OperandBase* Softmax(OperandBase*);
        OperandBase* Transpose(OperandBase*, TransposeOptions const* options);
        ModelBase* CreateModel(NamedOperandsBase const* named_operands);

      private:
        // Create concrete model.
        virtual ModelBase* CreateModelImpl() = 0;

        // Topological sort of nodes needed to compute rootNodes
        std::vector<const OperandBase*> TopologicalSort(std::vector<const OperandBase*>& rootNodes);
    };

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_MODEL_BUILDER_H_
