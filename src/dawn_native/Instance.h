#ifndef WEBNN_NATIVE_INSTANCE_H_
#define WEBNN_NATIVE_INSTANCE_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

namespace dawn_native {
  class InstanceBase : public RefCounted {
   public:
    InstanceBase() = default;
    virtual ~InstanceBase() = default;

    static InstanceBase* Create(const InstanceDescriptor* descriptor = nullptr) { return nullptr; }
  };
}

#endif  // WEBNN_NATIVE_INSTANCE_H_