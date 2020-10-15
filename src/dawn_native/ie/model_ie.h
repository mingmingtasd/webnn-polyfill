#ifndef WEBNN_NATIVE_IE_MODEL_IE_H_
#define WEBNN_NATIVE_IE_MODEL_IE_H_

#include <map>

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ie/ienn/src/ie_nn_c_api.h"

namespace dawn_native {

namespace ie {

class Model : public ModelBase {
public:
  Model(NamedOperand const *named_operands, size_t size);
  ~Model() override = default;

  OperandBase *GetNamedOperand(std::string name);

  void AddConstant(OperandBase *constant, OperandDescriptor const *desc,
                   void const *value, size_t size) override;
  void AddInput(OperandBase *input, const std::string name,
                OperandDescriptor const *desc) override;
  void AddMatMul(OperandBase *matmul, OperandBase *a, OperandBase *b) override;

  void BuildNeuralNetworkModel(OperandBase *root);
  void CompileImpl(WNNCompileCallback callback,
                   CompilationOptions const *options) override;

  ie_model_t *GetInferenceEngineModel();

private:
  void AddOutput(OperandBase *ouput);
  void Finish();

  ie_model_t *ie_model_;

  std::map<std::string, OperandBase *> named_operands_;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_MODEL_IE_H_