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
class Binary;
class Conv2d;
class Pool2d;
class Reshape;
class Transpose;
class Unary;
} // namespace op

class ModelBase : public RefCounted {
public:
  ModelBase() = default;
  virtual ~ModelBase() = default;

  // Dawn API
  void Compile(WNNCompileCallback callback, void *userdata,
               CompilationOptions const *options);

  virtual void AddConstant(const op::Constant *constant) = 0;
  virtual void AddInput(const op::Input *input) = 0;
  virtual void AddOutput(const OperandBase* output) = 0;
  virtual void AddBinary(const op::Binary *binary) = 0;
  virtual void AddConv2d(const op::Conv2d *conv2d) = 0;
  virtual void AddPool2d(const op::Pool2d *pool2d) = 0;
  virtual void AddReshape(const op::Reshape *relu) = 0;
  virtual void AddTranspose(const op::Transpose *transpose) = 0;
  virtual void AddUnary(const op::Unary *unary) = 0;

private:
  virtual void CompileImpl(WNNCompileCallback callback, void *userdata,
                           CompilationOptions const *options) = 0;
};
} // namespace dawn_native

#endif  // WEBNN_NATIVE_MODEL_H_