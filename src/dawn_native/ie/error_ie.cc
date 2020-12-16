
#include "dawn_native/ie/error_ie.h"

#include <sstream>
#include <string>

namespace dawn_native {

namespace ie {

MaybeError CheckStatusCodeImpl(IEStatusCode code, const char *context) {
  if (code == IEStatusCode::OK) {
    return {};
  }

  std::ostringstream messageStream;
  messageStream << context << " failed with status code " << code;

  return DAWN_INTERNAL_ERROR(messageStream.str());
}

} // namespace ie

} // namespace dawn_native
