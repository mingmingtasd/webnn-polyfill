#ifndef WEBNN_NATIVE_DEVICE_H_
#define WEBNN_NATIVE_DEVICE_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

namespace dawn_native {
  class DeviceBase : public RefCounted {
   public:
    DeviceBase() = default;
    virtual ~DeviceBase() = default;
  };
}

#endif  // WEBNN_NATIVE_DEVICE_H_