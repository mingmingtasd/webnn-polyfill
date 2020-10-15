
#include "dawn_native/Operand.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {

void CompilationBase::Compute(InputsBase *inputs, WNNComputeCallback callback,
                              OutputsBase *outputs) {
  ComputeImpl(inputs, callback, outputs);
}

} // namespace dawn_native
