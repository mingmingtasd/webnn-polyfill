#ifndef WEBNN_NATIVE_IE_COMPILATION_IE_H_
#define WEBNN_NATIVE_IE_COMPILATION_IE_H_

#include "dawn_native/Compilation.h"
#include "dawn_native/ie/ienn/src/ie_nn_c_api.h"
#include "dawn_native/ie/model_ie.h"

namespace dawn_native {

namespace ie {

class Compilation : public CompilationBase {
public:
  Compilation(Ref<Model> model);
  ~Compilation() override;

  void Compile(WNNCompileCallback callback,
                void *userdata, CompilationOptions const *options);

private:
  void ComputeImpl(NamedInputsBase *inputs, WNNComputeCallback callback,
                   void *userdata, NamedOutputsBase *outputs) override;

  Ref<Model> model_;
  ie_compilation_t *ie_compilation_;

  // Hold those variable to async compute.
  void CompletedCallback();
  ie_complete_call_back_t ie_callback_;
  WNNComputeCallback callback_;
  void *user_data_;
  NamedOutputsBase *outputs_;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_COMPILATION_IE_H_