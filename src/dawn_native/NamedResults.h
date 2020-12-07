#ifndef WEBNN_NATIVE_NAMED_RESULTS_H_
#define WEBNN_NATIVE_NAMED_RESULTS_H_

#include <map>
#include <string>

#include "dawn_native/NamedRecords.h"
#include "dawn_native/Result.h"

namespace dawn_native {

class NamedResultsBase : public NamedRecords<ResultBase> {};

}

#endif