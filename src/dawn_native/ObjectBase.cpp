
#include "dawn_native/ObjectBase.h"

namespace dawn_native {
    static constexpr uint64_t kErrorPayload = 0;
    static constexpr uint64_t kNotErrorPayload = 1;

    ObjectBase::ObjectBase(NeuralNetworkContextBase* context)
        : RefCounted(kNotErrorPayload), context_(context) {
    }

    ObjectBase::ObjectBase(NeuralNetworkContextBase* context, ErrorTag)
        : RefCounted(kErrorPayload), context_(context) {
    }

    NeuralNetworkContextBase* ObjectBase::GetContext() const {
        return context_;
    }

    bool ObjectBase::IsError() const {
        return GetRefCountPayload() == kErrorPayload;
    }

}  // namespace dawn_native
