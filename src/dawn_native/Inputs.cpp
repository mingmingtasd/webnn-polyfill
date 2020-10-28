
#include "dawn_native/Inputs.h"

#include <string>

namespace dawn_native {

void InputsBase::SetInput(char const *name, struct Input const *input) {
  inputs_[std::string(name)] = input;
}

std::map<std::string, Input const *> &InputsBase::GetInputs() {
  return inputs_;
}

} // namespace dawn_native
