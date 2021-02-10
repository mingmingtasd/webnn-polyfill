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

#ifndef WEBNN_NATIVE_IE_MODEL_IE_H_
#define WEBNN_NATIVE_IE_MODEL_IE_H_

#include <map>
#include <set>

#include "webnn_native/Error.h"
#include "webnn_native/Model.h"
#include "webnn_native/Operand.h"
#include "webnn_native/openvino/ModelBuilderIE.h"
#include "webnn_native/openvino/ienn/src/ie_nn_c_api.h"
#include "webnn_native/ops/Binary.h"
#include "webnn_native/ops/Constant.h"
#include "webnn_native/ops/Conv2d.h"
#include "webnn_native/ops/Input.h"
#include "webnn_native/ops/Pool2d.h"
#include "webnn_native/ops/Reshape.h"
#include "webnn_native/ops/Transpose.h"
#include "webnn_native/ops/Unary.h"

namespace webnn_native { namespace ie {

    class Model : public ModelBase {
      public:
        explicit Model(ModelBuilder* model_builder);
        ~Model() override;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* ouput) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReshape(const op::Reshape* relu) override;
        virtual MaybeError AddTranspose(const op::Transpose* transpose) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError Finish() override;

        ie_model_t* GetInferenceEngineModel();
        size_t GetOutputsNumber();
        std::string GetOutputId(size_t index);

        friend class Compilation;

      private:
        void CompileImpl(WebnnCompileCallback callback,
                         void* userdata,
                         CompilationOptions const* options) override;

        ie_model_t* mIeModel;

        // Map the input name to IE internal id
        std::map<std::string, std::string> mInputIdMap;
        // Map the IE internal id to output name
        std::map<std::string, std::string> mOutputNameMap;
        // Map the operand to IE internal id
        std::map<const OperandBase*, std::string> mOperandIdMap;
    };

}}  // namespace webnn_native::ie

#endif  // WEBNN_NATIVE_IE_MODEL_IE_H_
