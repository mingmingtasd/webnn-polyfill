#ifndef WEBNN_NATIVE_DML_MODEL_DML_H_
#define WEBNN_NATIVE_DML_MODEL_DML_H_

#include <map>
#include <set>

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ops/binary.h"
#include "dawn_native/ops/constant.h"
#include "dawn_native/ops/conv2d.h"
#include "dawn_native/ops/input.h"
#include "dawn_native/ops/pool2d.h"
#include "dawn_native/ops/reshape.h"
#include "dawn_native/ops/transpose.h"
#include "dawn_native/ops/unary.h"

namespace dawn_native {

namespace dml {

class Model : public ModelBase {
public:
  Model();
  ~Model() override = default;

  virtual void AddConstant(const op::Constant *constant) override;
  virtual void AddInput(const op::Input *input) override;
  virtual void AddOutput(const OperandBase* output) override;
  virtual void AddBinary(const op::Binary *binary) override;
  virtual void AddConv2d(const op::Conv2d *conv2d) override;
  virtual void AddPool2d(const op::Pool2d *pool2d) override;
  virtual void AddReshape(const op::Reshape *relu) override;
  virtual void AddTranspose(const op::Transpose *transpose) override;
  virtual void AddUnary(const op::Unary *unary) override;

private:
  void CompileImpl(WNNCompileCallback callback, void *userdata,
                   CompilationOptions const *options) override;
};

} // namespace dml

} // namespace dawn_native

#endif // WEBNN_NATIVE_DML_MODEL_DML_H_
