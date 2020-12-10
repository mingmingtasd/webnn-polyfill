#include "dawn_native/dml/model_dml.h"

#include "common/Assert.h"

namespace dawn_native {
namespace dml {

Model::Model(NamedOperandsBase const *named_operands) {
  UNREACHABLE();
}

void Model::AddConstant(op::Constant *constant) {
  UNREACHABLE();
}
  
void Model::AddInput(op::Input *input) {
  UNREACHABLE();
}
  
void Model::AddBinary(op::Binary *binary) {
  UNREACHABLE();
}

void Model::AddConv2d(op::Conv2d *conv2d) {
  UNREACHABLE();
}
  
void Model::AddPool2d(op::Pool2d *pool2d) {
  UNREACHABLE();
}

void Model::AddReshape(op::Reshape *relu) {
  UNREACHABLE();
}

void Model::AddTranspose(op::Transpose *transpose) {
  UNREACHABLE();
}
  
void Model::AddUnary(op::Unary *unary) {
  UNREACHABLE();
}

void Model::CompileImpl(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  UNREACHABLE();
}

}  // namespace dml
}  // namespace dawn_native
