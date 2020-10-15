
#include "dawn_native/Outputs.h"

#include <string>

namespace dawn_native {

void OutputsBase::SetOutput(char const *name, struct Output const *output) {
  outputs_[std::string(name)] = output;
}

WNNOutput OutputsBase::GetOutput(char const *name) {
  Output const *output = outputs_[std::string(name)];
  WNNOutput wnn_output;
  wnn_output.buffer = output->buffer;
  wnn_output.size = output->size;
  wnn_output.dimensions = output->dimensions;
  wnn_output.dimensionsCount = output->dimensionsCount;
  return wnn_output;
}

std::map<std::string, Output const *> &OutputsBase::GetOutputs() {
  return outputs_;
}

} // namespace dawn_native
