#ifndef WEBNN_NATIVE_IE_MODEL_IE_H_
#define WEBNN_NATIVE_IE_MODEL_IE_H_

#include <map>
#include <set>

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ie/ienn/src/ie_nn_c_api.h"
#include "dawn_native/ops/binary.h"
#include "dawn_native/ops/constant.h"
#include "dawn_native/ops/input.h"
#include "dawn_native/ops/matmul.h"

namespace dawn_native {

namespace ie {

class Model : public ModelBase {
public:
  Model(NamedOperand const *named_operands, size_t size);
  ~Model() override = default;

  virtual void AddConstant(op::Constant *constant) override;
  virtual void AddInput(op::Input *input) override;
  virtual void AddMatMul(op::MatMul *mat_mul) override;
  virtual void AddBinary(op::Binary *binary) override;

  OperandBase *GetNamedOperand(std::string name);
  ie_model_t *GetInferenceEngineModel();

private:
  void CompileImpl(WNNCompileCallback callback,
                   CompilationOptions const *options) override;
  void BuildNeuralNetworkModel(OperandBase *root);
  void AddOutput(OperandBase *ouput);
  void Finish();

  ie_model_t *ie_model_;

  std::set<OperandBase *> traversalled_;
  std::map<std::string, OperandBase *> named_operands_;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_MODEL_IE_H_