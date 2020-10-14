#ifndef WEBNN_NATIVE_COMPILATION_H_
#define WEBNN_NATIVE_COMPILATION_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

#include "dawn_native/dawn_platform.h"

#include "dawn_native/Outputs.h"

namespace dawn_native {
  class CompilationBase : public RefCounted {
   public:
    CompilationBase() = default;
    virtual ~CompilationBase() = default;

    void Compute(InputsBase* inputs, WNNComputeCallback callback, OutputsBase* outputs = nullptr) {
      callback(reinterpret_cast<WNNOutputs>(new OutputsBase()));
    }
  };
}

#endif  // WEBNN_NATIVE_COMPILATION_H_