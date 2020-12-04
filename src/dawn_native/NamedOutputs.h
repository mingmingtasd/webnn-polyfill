#ifndef WEBNN_NATIVE_NAMED_OUTPUTS_H_
#define WEBNN_NATIVE_NAMED_OUTPUTS_H_

#include <map>
#include <string>

#include "dawn_native/NamedRecords.h"

namespace dawn_native {

class NamedOutputsBase : public NamedRecords<Output> {};

}

#endif