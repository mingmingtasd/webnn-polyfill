#ifndef WEBNN_NATIVE_IE_MODEL_IE_H_
#define WEBNN_NATIVE_IE_MODEL_IE_H_

#include <map>
#include <set>

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ie/ienn/src/ie_nn_c_api.h"
#include "dawn_native/ops/binary.h"
#include "dawn_native/ops/constant.h"
#include "dawn_native/ops/conv2d.h"
#include "dawn_native/ops/input.h"
#include "dawn_native/ops/matmul.h"
#include "dawn_native/ops/relu.h"
#include "dawn_native/ops/reshape.h"
#include "dawn_native/ops/softmax.h"
#include "dawn_native/ops/transpose.h"

namespace dawn_native {

namespace ie {

class Model : public ModelBase {
public:
  Model(NamedOperandsBase const *named_operands);
  ~Model() override = default;

  virtual void AddConstant(op::Constant *constant) override;
  virtual void AddInput(op::Input *input) override;
  virtual void AddMatMul(op::MatMul *mat_mul) override;
  virtual void AddBinary(op::Binary *binary) override;
  virtual void AddConv2d(op::Conv2d *conv2d) override;
  virtual void AddPool2d(op::Pool2d *pool2d) override;
  virtual void AddRelu(op::Relu *relu) override;
  virtual void AddReshape(op::Reshape *relu) override;
  virtual void AddSoftmax(op::Softmax *softmax) override;
  virtual void AddTranspose(op::Transpose *transpose) override;

  const OperandBase *GetNamedOperand(std::string name);
  ie_model_t *GetInferenceEngineModel();
  size_t GetOutputsNumber();
  std::string GetOutputName(size_t index);
  const std::string& GetUserName(const std::string& name);

private:
  void CompileImpl(WNNCompileCallback callback, void *userdata,
                   CompilationOptions const *options) override;
  void BuildNeuralNetworkModel(const OperandBase *root);
  void AddOutput(const OperandBase *ouput);
  void Finish();

  ie_model_t *ie_model_;

  std::set<const OperandBase *> traversalled_;
  std::map<std::string, const OperandBase*> named_operands_;
  std::map<std::string, std::string> user_name_map_;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_MODEL_IE_H_