
#include "dawn_native/Operand.h"

#include "common/Assert.h"
#include "common/Log.h"
#include "dawn_native/NamedResults.h"

namespace dawn_native {

void CompilationBase::Compute(NamedInputsBase *inputs,
                              WNNComputeCallback callback,
                              void *userdata, NamedOutputsBase *outputs) {
  ComputeImpl(inputs, callback, userdata, outputs);
}

} // namespace dawn_native
