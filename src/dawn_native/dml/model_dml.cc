#include "dawn_native/dml/model_dml.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {
namespace dml {

Model::Model() {
  DAWN_DEBUG();
}

void Model::AddConstant(const op::Constant *constant) {
  DAWN_DEBUG();
}
  
void Model::AddInput(const op::Input *input) {
  DAWN_DEBUG();
}

void Model::AddOutput(const OperandBase* output) {
  DAWN_DEBUG();
}
  
void Model::AddBinary(const op::Binary *binary) {
  DAWN_DEBUG();
}

void Model::AddConv2d(const op::Conv2d *conv2d) {
  DAWN_DEBUG();
}
  
void Model::AddPool2d(const op::Pool2d *pool2d) {
  DAWN_DEBUG();
}

void Model::AddReshape(const op::Reshape *relu) {
  DAWN_DEBUG();
}

void Model::AddTranspose(const op::Transpose *transpose) {
  DAWN_DEBUG();
}
  
void Model::AddUnary(const op::Unary *unary) {
  DAWN_DEBUG();
}

void Model::CompileImpl(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  DAWN_DEBUG();
}

}  // namespace dml
}  // namespace dawn_native
