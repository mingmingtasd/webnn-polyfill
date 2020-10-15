
#include "dawn_native/Inputs.h"

#include <string>

namespace dawn_native {

void InputsBase::SetInput(char const *name, struct Input const *input) {
  // float_inputs_.insert({std::string(name), 1.0});
  float_inputs_[1] = 1.0;
  inputs_[std::string(name)] = input;
}

std::map<std::string, Input const *> &InputsBase::GetInputs() {
  return inputs_;
}

} // namespace dawn_native
