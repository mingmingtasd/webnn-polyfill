#ifndef WEBNN_NATIVE_DML_COMPILATION_DML_H_
#define WEBNN_NATIVE_DML_COMPILATION_DML_H_

#include "dawn_native/Compilation.h"
#include "dawn_native/dml/ModelDML.h"

namespace pydml {
    struct CompiledModel;
}

namespace dawn_native { namespace dml {

    class Compilation : public CompilationBase {
      public:
        explicit Compilation(const Ref<Model>& model);
        ~Compilation() override = default;

      private:
        void ComputeImpl(NamedInputsBase* inputs,
                         WNNComputeCallback callback,
                         void* userdata,
                         NamedOutputsBase* outputs = nullptr) override;

        Ref<Model> model_;
        std::unique_ptr<pydml::CompiledModel> compiled_model_;
    };

}}  // namespace dawn_native::dml

#endif  // WEBNN_NATIVE_DML_COMPILATION_DML_H_