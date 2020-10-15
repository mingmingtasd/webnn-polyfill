#ifndef WEBNN_NATIVE_OUTPUTS_H_
#define WEBNN_NATIVE_OUTPUTS_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

#include "dawn_native/dawn_platform.h"

namespace dawn_native {
  class OutputsBase : public RefCounted {
   public:
    OutputsBase() = default;
    virtual ~OutputsBase() = default;

    void SetOutput(char const * name, Output const * output) {}
    WNNOutput GetOutput(char const * name) {return {};}
  };
}

#endif