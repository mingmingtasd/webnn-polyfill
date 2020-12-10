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
  Model(NamedOperandsBase const *named_operands);
  ~Model() override = default;

  virtual void AddConstant(op::Constant *constant) override;
  virtual void AddInput(op::Input *input) override;
  virtual void AddBinary(op::Binary *binary) override;
  virtual void AddConv2d(op::Conv2d *conv2d) override;
  virtual void AddPool2d(op::Pool2d *pool2d) override;
  virtual void AddReshape(op::Reshape *relu) override;
  virtual void AddTranspose(op::Transpose *transpose) override;
  virtual void AddUnary(op::Unary *unary) override;

private:
  void CompileImpl(WNNCompileCallback callback, void *userdata,
                   CompilationOptions const *options) override;
};

} // namespace dml

} // namespace dawn_native

#endif // WEBNN_NATIVE_DML_MODEL_DML_H_
