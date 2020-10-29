#ifndef WEBNN_NATIVE_INPUTS_H_
#define WEBNN_NATIVE_INPUTS_H_

#include <map>

#include "common/RefCounted.h"

namespace dawn_native {

class InputsBase : public RefCounted {
public:
  InputsBase() = default;
  virtual ~InputsBase() = default;

  // DAWN API
  void SetInput(char const *name, struct Input const *input);
  std::map<std::string, Input const *> &GetInputs();

private:
  std::map<std::string, Input const *> inputs_;
};
}

#endif