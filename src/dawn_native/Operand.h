#ifndef WEBNN_NATIVE_OPERAND_H_
#define WEBNN_NATIVE_OPERAND_H_

#include <string>

#include "common/RefCounted.h"
#include "dawn_native/Forward.h"
#include "dawn_native/Model.h"

#include "dawn_native/dawn_platform.h"

namespace dawn_native {

class OperandBase : public RefCounted {
public:
  OperandBase() = default;
  virtual ~OperandBase() = default;

  // First/NextInput are used for getting inputs when traversaling model tree.
  virtual Ref<OperandBase> FirstInput() const;
  virtual Ref<OperandBase> NextInput();
  void SetTraversal(bool traversal);
  bool Traversal();

  virtual void AddOperand(ModelBase *model);
  void SetName(std::string name);
  std::string GetName();
  // Use a chained struct to hold the identification of the operand, name for
  // Inference Engine, index for Android NNAPI.

private:
  // The traversalled operand doesn't traversal again.
  bool traversalled_;
  // The operand name.
  std::string name_;
};
}

#endif  // WEBNN_NATIVE_OPERAND_H_