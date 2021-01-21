#ifndef WEBNN_NATIVE_MODEL_BUILDER_H_
#define WEBNN_NATIVE_MODEL_BUILDER_H_

#include "common/RefCounted.h"
#include "dawn_native/Forward.h"
#include "dawn_native/NamedOperands.h"
#include "dawn_native/ObjectBase.h"
#include "dawn_native/dawn_platform.h"

#include <vector>

namespace dawn_native {

    class ModelBuilderBase : public ObjectBase {
      public:
        ModelBuilderBase(NeuralNetworkContextBase* context);
        virtual ~ModelBuilderBase() = default;

        // DAWN API
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

        // Topological sort of nodes needed to compute root_nodes
        std::vector<const OperandBase*> TopologicalSort(
            std::vector<const OperandBase*>& root_nodes);
    };

}  // namespace dawn_native

#endif  // WEBNN_NATIVE_MODEL_BUILDER_H_
