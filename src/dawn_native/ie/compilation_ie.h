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

  OutputsBase *ComputeImpl(InputsBase *inputs, WNNComputeCallback callback,
                           void *userdata,
                           OutputsBase *outputs = nullptr) override;
  void FreeUnusedData();

private:
  Ref<Model> model_;
  ie_compilation_t *ie_compilation_;
  // The outputs is used to hold buffer from Inference Engine.
  std::vector<Output *> outputs_;
};

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_COMPILATION_IE_H_