// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/unary.h"

#include "common/Log.h"
#include "dawn_native/Error.h"
#include "dawn_native/ops/utils.h"

namespace dawn_native {

namespace op {

std::string UnaryOpTypeToString(UnaryOpType type) {
  if (type == UnaryOpType::kRelu) {
    return "relu";
  } else if (type == UnaryOpType::kSoftmax) {
    return "softmax";
  }
  return std::to_string(type);
}

MaybeError Unary::ValidateAndInferTypes() {
  auto input = inputs_[0];
  type_ = input->Type();
  dimensions_ = input->Dimensions();

  DAWN_DEBUG() << " op type: " << UnaryOpTypeToString(OpType())
               << ", input.type: " << OperandTypeToString(input->Type())
               << ", input.dimensions: " << ShapeToString(input->Dimensions())
               << ", output.type: " << OperandTypeToString(type_)
               << ", output.dimensions: " << ShapeToString(dimensions_);
  return {};
}

}  // namespace op
}  // namespace dawn_native
