#ifndef WEBNN_NATIVE_NEURALNETWORKCONTEXT_H_
#define WEBNN_NATIVE_NEURALNETWORKCONTEXT_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

#include "dawn_native/dawn_platform.h"

namespace dawn_native {
  class NeuralNetworkContextBase : public RefCounted {
   public:
    NeuralNetworkContextBase() = default;
    virtual ~NeuralNetworkContextBase() = default;

    OperandBase* Constant(OperandDescriptor const * desc, void const * value, size_t offset, size_t size) {return nullptr;}
    OperandBase* Input(char const * name, OperandDescriptor const * desc) {return nullptr;}
    OperandBase* Matmul(OperandBase* a, OperandBase* b) {return nullptr;}

    ModelBase* CreateModel() {return nullptr;}
  };
}

#endif  // WEBNN_NATIVE_NEURALNETWORKCONTEXT_H_