
#include "dawn_native/Model.h"

#include <string>

#include "common/Assert.h"
#include "common/RefCounted.h"

namespace dawn_native {

ModelBase::ModelBase(ModelBuilderBase *model_builder)
    : ObjectBase(model_builder->GetContext()) {}

void ModelBase::Compile(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  if (DAWN_UNLIKELY(this->IsError())) {
    callback(WNNCompileStatus_Error, nullptr, "Object is an error", userdata);
    return;
  }
  CompileImpl(callback, userdata, options);
}

ModelBase::ModelBase(ModelBuilderBase *model_builder, ObjectBase::ErrorTag tag)
    : ObjectBase(model_builder->GetContext(), tag) {}

// static
ModelBase *ModelBase::MakeError(ModelBuilderBase *model_builder) {
  return new ModelBase(model_builder, ObjectBase::kError);
}

MaybeError ModelBase::AddConstant(const op::Constant *constant) {
  UNREACHABLE();
}

MaybeError ModelBase::AddInput(const op::Input *input) { UNREACHABLE(); }

MaybeError ModelBase::AddOutput(const std::string &name,
                                const OperandBase *output) {
  UNREACHABLE();
}

MaybeError ModelBase::AddBinary(const op::Binary *binary) { UNREACHABLE(); }

MaybeError ModelBase::AddConv2d(const op::Conv2d *conv2d) { UNREACHABLE(); }

MaybeError ModelBase::AddPool2d(const op::Pool2d *pool2d) { UNREACHABLE(); }

MaybeError ModelBase::AddReshape(const op::Reshape *relu) { UNREACHABLE(); }

MaybeError ModelBase::AddTranspose(const op::Transpose *transpose) {
  UNREACHABLE();
}

MaybeError ModelBase::AddUnary(const op::Unary *unary) { UNREACHABLE(); }

MaybeError ModelBase::Finish() { UNREACHABLE(); }

void ModelBase::CompileImpl(WNNCompileCallback callback, void *userdata,
                            CompilationOptions const *options) {
  UNREACHABLE();
}

} // namespace dawn_native
