// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef WEBNN_NATIVE_IE_COMPILATION_IE_H_
#define WEBNN_NATIVE_IE_COMPILATION_IE_H_

#include "dawn_native/Compilation.h"
#include "dawn_native/ie/ModelIE.h"
#include "dawn_native/ie/ienn/src/ie_nn_c_api.h"

namespace dawn_native { namespace ie {

    class Compilation : public CompilationBase {
      public:
        Compilation(Ref<Model> model);
        ~Compilation() override;

        void Compile(WNNCompileCallback callback,
                     void* userdata,
                     CompilationOptions const* options);

      private:
        void ComputeImpl(NamedInputsBase* inputs,
                         WNNComputeCallback callback,
                         void* userdata,
                         NamedOutputsBase* outputs) override;

        Ref<Model> model_;
        ie_compilation_t* ie_compilation_;

        // Hold those variable to async compute.
        void CompletedCallback();
        ie_complete_call_back_t ie_callback_;
        WNNComputeCallback callback_;
        void* user_data_;
        NamedOutputsBase* outputs_;
    };

}}  // namespace dawn_native::ie

#endif  // WEBNN_NATIVE_IE_COMPILATION_IE_H_
