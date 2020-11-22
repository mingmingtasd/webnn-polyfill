#ifndef WEBNN_NATIVE_OUTPUTS_H_
#define WEBNN_NATIVE_OUTPUTS_H_

#include <map>
#include <string>

#include "common/RefCounted.h"
#include "dawn_native/Forward.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

class OutputsBase : public RefCounted {
public:
  OutputsBase() = default;
  virtual ~OutputsBase() = default;

  // DAWN API
  void SetOutput(char const *name, struct Output const *output);
  WNNOutput GetOutput(char const *name);
  size_t GetOutputsNumber();
  WNNOutput GetOutputWithIndex(size_t index);

  std::map<std::string, Output const *> &GetOutputs();

private:
  std::map<std::string, Output const *> outputs_;
};
}

#endif