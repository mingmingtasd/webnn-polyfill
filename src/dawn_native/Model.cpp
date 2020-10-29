
#include "dawn_native/Model.h"

#include <string>

#include "common/Assert.h"
#include "common/RefCounted.h"

namespace dawn_native {

void ModelBase::Compile(WNNCompileCallback callback,
                        CompilationOptions const *options) {
  CompileImpl(callback, options);
}

void ModelBase::CompileImpl(WNNCompileCallback callback,
                            CompilationOptions const *options) {
  UNREACHABLE();
}

} // namespace dawn_native
