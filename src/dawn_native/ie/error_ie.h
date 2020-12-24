#ifndef WEBNN_NATIVE_IE_ERROR_IE_H_
#define WEBNN_NATIVE_IE_ERROR_IE_H_

#include "dawn_native/Error.h"
#include "dawn_native/ie/ienn/src/ie_nn_c_api.h"

namespace dawn_native {

namespace ie {

MaybeError CheckStatusCodeImpl(IEStatusCode code, const char *context);

#define CheckStatusCode(code, context) CheckStatusCodeImpl(code, context)

} // namespace ie

} // namespace dawn_native

#endif // WEBNN_NATIVE_IE_ERROR_IE_H_