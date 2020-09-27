#ifndef WEBNN_NATIVE_MODEL_H_
#define WEBNN_NATIVE_MODEL_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

#include "dawn_native/dawn_platform.h"

#include "dawn_native/Compilation.h"

namespace dawn_native {
  class ModelBase : public RefCounted {
   public:
    ModelBase() = default;
    virtual ~ModelBase() = default;

    void Compile(WGPUCompileCallback callback, CompilationOptions const * options) {
      callback(reinterpret_cast<WGPUCompilation>(new CompilationBase()));
    }
  };
}

#endif  // WEBNN_NATIVE_MODEL_H_