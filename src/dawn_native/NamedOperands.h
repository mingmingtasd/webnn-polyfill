#ifndef WEBNN_NATIVE_NAMED_OPERANDS_H_
#define WEBNN_NATIVE_NAMED_OPERANDS_H_

#include <map>
#include <string>

#include "dawn_native/NamedRecords.h"

namespace dawn_native {

    class NamedOperandsBase : public NamedRecords<OperandBase> {};

}  // namespace dawn_native

#endif