
#include "dawn_native/Model.h"

#include <string>

#include "common/Assert.h"
#include "common/RefCounted.h"

namespace dawn_native {

void ModelBase::Compile(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  CompileImpl(callback, userdata, options);
}

void ModelBase::CompileImpl(WNNCompileCallback callback, void *userdata,
                            CompilationOptions const *options) {
  UNREACHABLE();
}

} // namespace dawn_native
