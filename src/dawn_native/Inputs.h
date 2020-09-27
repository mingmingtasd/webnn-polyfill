#ifndef WEBNN_NATIVE_INPUTS_H_
#define WEBNN_NATIVE_INPUTS_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

#include "dawn_native/dawn_platform.h"

namespace dawn_native {
  class InputsBase : public RefCounted {
   public:
    InputsBase() = default;
    virtual ~InputsBase() = default;

    void SetInput(char const * name, Input const * input) {}
  };
}

#endif