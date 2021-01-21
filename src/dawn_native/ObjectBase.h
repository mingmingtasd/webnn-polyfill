#ifndef WEBNN_NATIVE_OBJECT_BASE_H_
#define WEBNN_NATIVE_OBJECT_BASE_H_

#include "dawn_native/NeuralNetworkContext.h"

namespace dawn_native {

    class ObjectBase : public RefCounted {
      public:
        struct ErrorTag {};
        static constexpr ErrorTag kError = {};

        explicit ObjectBase(NeuralNetworkContextBase* context);
        ObjectBase(NeuralNetworkContextBase* context, ErrorTag tag);

        NeuralNetworkContextBase* GetContext() const;
        bool IsError() const;

      protected:
        ~ObjectBase() override = default;

      private:
        NeuralNetworkContextBase* context_;
    };

}  // namespace dawn_native

#endif  // WEBNN_NATIVE_OBJECT_BASE_H_