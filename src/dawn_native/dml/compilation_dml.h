#ifndef WEBNN_NATIVE_DML_COMPILATION_DML_H_
#define WEBNN_NATIVE_DML_COMPILATION_DML_H_

#include "dawn_native/Compilation.h"
#include "dawn_native/dml/model_dml.h"

namespace dawn_native {

namespace dml {

class Compilation : public CompilationBase {
public:
  explicit Compilation(const Model* model);
  ~Compilation() override;

private:
  void ComputeImpl(NamedInputsBase *inputs, WNNComputeCallback callback,
                   void *userdata,
                   NamedOutputsBase *outputs = nullptr) override;
};

} // namespace dml

} // namespace dawn_native

#endif // WEBNN_NATIVE_DML_COMPILATION_DML_H_