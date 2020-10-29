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
  std::vector<Ref<OperandBase>> &Inputs();
  // Add the operand to model for specific backend.
  virtual void AddToModel(ModelBase *model);
  // The name is uniquely identifies getting from native api.
  void SetName(std::string name);
  std::string GetName();

private:
  // the inputs of operand.
  std::vector<Ref<OperandBase>> inputs_;
  // The operand name.
  std::string name_;
};
}

#endif  // WEBNN_NATIVE_OPERAND_H_