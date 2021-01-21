#ifndef WEBNN_NATIVE_MODEL_H_
#define WEBNN_NATIVE_MODEL_H_

#include "common/RefCounted.h"
#include "dawn_native/Compilation.h"
#include "dawn_native/Error.h"
#include "dawn_native/Forward.h"
#include "dawn_native/ModelBuilder.h"
#include "dawn_native/ObjectBase.h"
#include "dawn_native/Operand.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

    namespace op {
        class Constant;
        class Input;
        class Binary;
        class Conv2d;
        class Pool2d;
        class Reshape;
        class Transpose;
        class Unary;
    }  // namespace op

    class ModelBase : public ObjectBase {
      public:
        explicit ModelBase(ModelBuilderBase* model_builder);
        virtual ~ModelBase() = default;

        // static
        static ModelBase* MakeError(ModelBuilderBase* model_builder);

        // Dawn API
        void Compile(WNNCompileCallback callback,
                     void* userdata,
                     CompilationOptions const* options);

        virtual MaybeError AddConstant(const op::Constant* constant);
        virtual MaybeError AddInput(const op::Input* input);
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* output);
        virtual MaybeError AddBinary(const op::Binary* binary);
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d);
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d);
        virtual MaybeError AddReshape(const op::Reshape* relu);
        virtual MaybeError AddTranspose(const op::Transpose* transpose);
        virtual MaybeError AddUnary(const op::Unary* unary);
        virtual MaybeError Finish();

      private:
        ModelBase(ModelBuilderBase* model_builder, ObjectBase::ErrorTag tag);
        virtual void CompileImpl(WNNCompileCallback callback,
                                 void* userdata,
                                 CompilationOptions const* options);
    };
}  // namespace dawn_native

#endif  // WEBNN_NATIVE_MODEL_H_