
#include "dawn_native/Operand.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {

OutputsBase *CompilationBase::Compute(InputsBase *inputs,
                                      WNNComputeCallback callback,
                                      void *userdata, OutputsBase *outputs) {
  return ComputeImpl(inputs, callback, userdata, outputs);
}

} // namespace dawn_native
