#include "dawn_native/ops/constant.h"

#include "common/Log.h"
#include "dawn_native/Error.h"
#include "dawn_native/ops/utils.h"

namespace dawn_native {

namespace op {

MaybeError Constant::ValidateAndInferTypes() {
  DAWN_DEBUG() << " constant.type: " << OperandTypeToString(type_)
               << ", constant.dimensions: " << ShapeToString(dimensions_);
  return {};
}

}  // namespace op
}  // namespace dawn_native
