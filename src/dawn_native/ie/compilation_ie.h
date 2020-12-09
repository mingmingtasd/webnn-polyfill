#ifndef WEBNN_NATIVE_IE_COMPILATION_IE_H_
#define WEBNN_NATIVE_IE_COMPILATION_IE_H_

#include "dawn_native/Compilation.h"
#include "dawn_native/ie/ienn/src/ie_nn_c_api.h"
#include "dawn_native/ie/model_ie.h"

namespace dawn_native {

namespace ie {

class Compilation : public CompilationBase {
public:
  explicit Compilation(Ref<Model> model);
  ~Compilation() override;

  void ComputeImpl(NamedInputsBase *inputs, WNNComputeCallback callback,
                   void *userdata,
                   NamedOutputsBase *outputs = nullptr) override;

private:
  Ref<Model> model_;
  ie_compilation_t *ie_compilation_;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_COMPILATION_IE_H_