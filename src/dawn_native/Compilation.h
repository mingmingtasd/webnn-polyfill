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

  // Dawn API
  void Compute(InputsBase *inputs, WNNComputeCallback callback, void *userdata,
               OutputsBase *outputs = nullptr);

private:
  virtual void ComputeImpl(InputsBase *inputs, WNNComputeCallback callback,
                           void *userdata, OutputsBase *outputs = nullptr) = 0;
};
}

#endif  // WEBNN_NATIVE_COMPILATION_H_