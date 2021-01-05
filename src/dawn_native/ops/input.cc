#include "dawn_native/ops/input.h"

#include "common/Log.h"
#include "dawn_native/Error.h"
#include "dawn_native/ops/utils.h"

namespace dawn_native {

namespace op {

MaybeError Input::ValidateAndInferTypes() {
  DAWN_DEBUG() << " input.type: " << OperandTypeToString(type_)
               << ", input.dimensions: " << ShapeToString(dimensions_);
  return {};
}

}  // namespace op
}  // namespace dawn_native
