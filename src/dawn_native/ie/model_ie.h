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
#include "dawn_native/ops/pool2d.h"
#include "dawn_native/ops/reshape.h"
#include "dawn_native/ops/transpose.h"
#include "dawn_native/ops/unary.h"

namespace dawn_native {

namespace ie {

class Model : public ModelBase {
public:
  Model();
  ~Model() override = default;

  virtual void AddConstant(const op::Constant *constant) override;
  virtual void AddInput(const op::Input *input) override;
  virtual void AddOutput(const std::string&name, const OperandBase *ouput) override;
  virtual void AddBinary(const op::Binary *binary) override;
  virtual void AddConv2d(const op::Conv2d *conv2d) override;
  virtual void AddPool2d(const op::Pool2d *pool2d) override;
  virtual void AddReshape(const op::Reshape *relu) override;
  virtual void AddTranspose(const op::Transpose *transpose) override;
  virtual void AddUnary(const op::Unary *unary) override;
  virtual void Finish() override;

  ie_model_t *GetInferenceEngineModel();
  size_t GetOutputsNumber();
  std::string GetOutputId(size_t index);

  friend class Compilation;
private:
  void CompileImpl(WNNCompileCallback callback, void *userdata,
                   CompilationOptions const *options) override;  

  ie_model_t *ie_model_;

  // Map the input name to IE internal id
  std::map<std::string, std::string> input_id_map_;
  // Map the IE internal id to output name
  std::map<std::string, std::string> output_name_map_;
  // Map the operand to IE internal id
  std::map<const OperandBase*, std::string> operand_id_map_;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_MODEL_IE_H_