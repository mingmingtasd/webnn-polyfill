#ifndef WEBNN_NATIVE_NAMED_INPUTS_H_
#define WEBNN_NATIVE_NAMED_INPUTS_H_

#include <map>
#include <string>

#include "dawn_native/NamedRecords.h"

namespace dawn_native {

class NamedInputsBase : public NamedRecords<Input> {};

}

#endif