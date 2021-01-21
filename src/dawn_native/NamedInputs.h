#ifndef WEBNN_NATIVE_NAMED_INPUTS_H_
#define WEBNN_NATIVE_NAMED_INPUTS_H_

#include <map>
#include <string>

#include "dawn_native/NamedRecords.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

    class NamedInputsBase : public NamedRecords<Input> {};

}  // namespace dawn_native

#endif