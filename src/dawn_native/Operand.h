#ifndef WEBNN_NATIVE_OPERAND_H_
#define WEBNN_NATIVE_OPERAND_H_

#include "dawn_native/Forward.h"
#include "common/RefCounted.h"

#include "dawn_native/dawn_platform.h"

namespace dawn_native {
  class OperandBase : public RefCounted {
   public:
    OperandBase() = default;
    virtual ~OperandBase() = default;
  };
}

#endif  // WEBNN_NATIVE_OPERAND_H_