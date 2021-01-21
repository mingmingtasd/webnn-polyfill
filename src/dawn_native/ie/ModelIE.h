#ifndef WEBNN_NATIVE_IE_MODEL_IE_H_
#define WEBNN_NATIVE_IE_MODEL_IE_H_

#include <map>
#include <set>

#include "dawn_native/Error.h"
#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ie/ModelBuilderIE.h"
#include "dawn_native/ie/ienn/src/ie_nn_c_api.h"
#include "dawn_native/ops/Binary.h"
#include "dawn_native/ops/Constant.h"
#include "dawn_native/ops/Conv2d.h"
#include "dawn_native/ops/Input.h"
#include "dawn_native/ops/Pool2d.h"
#include "dawn_native/ops/Reshape.h"
#include "dawn_native/ops/Transpose.h"
#include "dawn_native/ops/Unary.h"

namespace dawn_native { namespace ie {

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
        void CompileImpl(WNNCompileCallback callback,
                         void* userdata,
                         CompilationOptions const* options) override;

        ie_model_t* ie_model_;

        // Map the input name to IE internal id
        std::map<std::string, std::string> input_id_map_;
        // Map the IE internal id to output name
        std::map<std::string, std::string> output_name_map_;
        // Map the operand to IE internal id
        std::map<const OperandBase*, std::string> operand_id_map_;
    };

}}  // namespace dawn_native::ie

#endif  // WEBNN_NATIVE_IE_MODEL_IE_H_