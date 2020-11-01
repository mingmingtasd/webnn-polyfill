#ifndef WEBNN_NATIVE_MODEL_H_
#define WEBNN_NATIVE_MODEL_H_

#include "common/RefCounted.h"
#include "dawn_native/Compilation.h"
#include "dawn_native/Forward.h"
#include "dawn_native/Operand.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

namespace op {
class Constant;
class Input;
class MatMul;
class Binary;
class Conv2d;
class Pool2d;
class Relu;
} // namespace op

class ModelBase : public RefCounted {
public:
  ModelBase() = default;
  virtual ~ModelBase() = default;

  // Dawn API
  void Compile(WNNCompileCallback callback, CompilationOptions const *options);

  virtual void AddConstant(op::Constant *constant) = 0;
  virtual void AddInput(op::Input *input) = 0;
  virtual void AddMatMul(op::MatMul *mat_mul) = 0;
  virtual void AddBinary(op::Binary *binary) = 0;
  virtual void AddConv2d(op::Conv2d *conv2d) = 0;
  virtual void AddPool2d(op::Pool2d *pool2d) = 0;
  virtual void AddRelu(op::Relu *relu) = 0;

private:
  virtual void CompileImpl(WNNCompileCallback callback,
                           CompilationOptions const *options) = 0;
};
} // namespace dawn_native

#endif  // WEBNN_NATIVE_MODEL_H_