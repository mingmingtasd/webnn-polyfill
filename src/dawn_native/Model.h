#ifndef WEBNN_NATIVE_MODEL_H_
#define WEBNN_NATIVE_MODEL_H_

#include "common/RefCounted.h"
#include "dawn_native/Compilation.h"
#include "dawn_native/Forward.h"
#include "dawn_native/Operand.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

class ModelBase : public RefCounted {
public:
  ModelBase() = default;
  virtual ~ModelBase() = default;

  void Compile(WNNCompileCallback callback, CompilationOptions const *options);

  virtual void AddConstant(OperandBase *, OperandDescriptor const *desc,
                           void const *value, size_t size) = 0;
  virtual void AddInput(OperandBase *, const std::string name,
                        OperandDescriptor const *desc) = 0;
  virtual void AddMatMul(OperandBase *, OperandBase *, OperandBase *) = 0;
  virtual void CompileImpl(WNNCompileCallback callback,
                           CompilationOptions const *options) = 0;
};
} // namespace dawn_native

#endif  // WEBNN_NATIVE_MODEL_H_