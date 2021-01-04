#ifndef WEBNN_NATIVE_OPERAND_H_
#define WEBNN_NATIVE_OPERAND_H_

#include <string>
#include <vector>

#include "dawn_native/Forward.h"
#include "dawn_native/Model.h"
#include "dawn_native/ObjectBase.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

class OperandBase : public ObjectBase {
public:
  explicit OperandBase(ModelBuilderBase *model_builder,
                       std::vector<Ref<OperandBase>> = {});
  virtual ~OperandBase() = default;

  // static
  static OperandBase *MakeError(ModelBuilderBase *model_builder);

  // Add the operand to model for specific backend.
  virtual MaybeError AddToModel(ModelBase *model) const { UNREACHABLE(); }
  // Validate the inputs and infer types and shapes.
  virtual MaybeError ValidateAndInferTypes() { UNREACHABLE(); }

  // It's used for getting inputs when traversaling model tree.
  const std::vector<Ref<OperandBase>> &Inputs() const { return inputs_; }
  wnn::OperandType Type() const { return type_; }
  const std::vector<int32_t>& Dimensions() const { return dimensions_; }
private:
  OperandBase(ModelBuilderBase *model_builder, ObjectBase::ErrorTag tag);

protected:
  // The inputs of operand.
  std::vector<Ref<OperandBase>> inputs_;
  // The operand type.
  wnn::OperandType type_;
  // The dimensions;
  std::vector<int32_t> dimensions_;
};
}

#endif  // WEBNN_NATIVE_OPERAND_H_