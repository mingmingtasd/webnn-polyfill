#ifndef WEBNN_NATIVE_DML_MODEL_DML_H_
#define WEBNN_NATIVE_DML_MODEL_DML_H_

#include <map>
#include <set>

#include "dawn_native/dml/ModelBuilderDML.h"
#include "dawn_native/dml/deps/src/precomp.h"
#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ops/Binary.h"
#include "dawn_native/ops/Constant.h"
#include "dawn_native/ops/Conv2d.h"
#include "dawn_native/ops/Input.h"
#include "dawn_native/ops/Pool2d.h"
#include "dawn_native/ops/Reshape.h"
#include "dawn_native/ops/Transpose.h"
#include "dawn_native/ops/Unary.h"

namespace dawn_native {

namespace dml {

std::string DmlTensorDimensionsToString(const ::dml::TensorDimensions&);
std::string DmlTensorDataTypeToString(DML_TENSOR_DATA_TYPE type);

class Model : public ModelBase {
public:
  explicit Model(ModelBuilder *model_builder);
  ~Model() override = default;

  virtual MaybeError AddConstant(const op::Constant *constant) override;
  virtual MaybeError AddInput(const op::Input *input) override;
  virtual MaybeError AddOutput(const std::string& name,
                               const OperandBase* output) override;
  virtual MaybeError AddBinary(const op::Binary *binary) override;
  virtual MaybeError AddConv2d(const op::Conv2d *conv2d) override;
  virtual MaybeError AddPool2d(const op::Pool2d *pool2d) override;
  virtual MaybeError AddReshape(const op::Reshape *relu) override;
  virtual MaybeError AddTranspose(const op::Transpose *transpose) override;
  virtual MaybeError AddUnary(const op::Unary *unary) override;
  virtual MaybeError Finish() override;

  friend class Compilation;
private:
  void CompileImpl(WNNCompileCallback callback, void *userdata,
                   CompilationOptions const *options) override;

  std::shared_ptr<::pydml::Device> device_;
  std::unique_ptr<::dml::Graph> graph_;
  std::map<const OperandBase*, ::dml::Expression> expressions_;
  std::vector<std::unique_ptr<::pydml::Binding>> bindings_;
  std::map<std::string, ::pydml::Binding*> inputs_;
  std::map<std::string, ::dml::Expression> outputs_;
};

} // namespace dml

} // namespace dawn_native

#endif // WEBNN_NATIVE_DML_MODEL_DML_H_
