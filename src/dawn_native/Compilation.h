#ifndef WEBNN_NATIVE_COMPILATION_H_
#define WEBNN_NATIVE_COMPILATION_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

#include "dawn_native/dawn_platform.h"

#include "dawn_native/NamedInputs.h"
#include "dawn_native/NamedOutputs.h"

namespace dawn_native {

class CompilationBase : public RefCounted {
public:
  CompilationBase() = default;
  virtual ~CompilationBase() = default;

  // Dawn API
  void Compute(NamedInputsBase *inputs, WNNComputeCallback callback,
               void *userdata, NamedOutputsBase *outputs = nullptr);

private:
  virtual void ComputeImpl(NamedInputsBase *inputs,
                           WNNComputeCallback callback, void *userdata,
                           NamedOutputsBase *outputs = nullptr) = 0;
};
}

#endif  // WEBNN_NATIVE_COMPILATION_H_