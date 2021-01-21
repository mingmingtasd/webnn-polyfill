
#include "dawn_native/ie/error_ie.h"

#include <sstream>
#include <string>

namespace dawn_native {

namespace ie {

MaybeError CheckStatusCodeImpl(IEStatusCode code, const char *context) {
  std::ostringstream error_message;
  error_message << context << " failed with status code " << code;

  switch (code) {
  case IEStatusCode::OK:
    break;
  case IEStatusCode::GENERAL_ERROR:
  case IEStatusCode::PARAMETER_MISMATCH:
  case IEStatusCode::NOT_FOUND:
  case IEStatusCode::OUT_OF_BOUNDS:
    return DAWN_VALIDATION_ERROR(error_message.str());
  default:
    return DAWN_INTERNAL_ERROR(error_message.str());
  }
  return {};
}

} // namespace ie

} // namespace dawn_native
