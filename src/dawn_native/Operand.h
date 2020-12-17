#ifndef WEBNN_NATIVE_OPERAND_H_
#define WEBNN_NATIVE_OPERAND_H_

#include <string>
#include <vector>

#include "common/RefCounted.h"
#include "dawn_native/Forward.h"
#include "dawn_native/Model.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

class OperandBase : public RefCounted {
public:
  explicit OperandBase(std::vector<Ref<OperandBase>>);
  virtual ~OperandBase() = default;

  // It's used for getting inputs when traversaling model tree.
  const std::vector<Ref<OperandBase>> &Inputs() const;
  // Add the operand to model for specific backend.
  virtual void AddToModel(ModelBase *model) const;

private:
  // the inputs of operand.
  std::vector<Ref<OperandBase>> inputs_;
};
}

#endif  // WEBNN_NATIVE_OPERAND_H_