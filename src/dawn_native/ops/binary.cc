// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/binary.h"

#include "common/Log.h"
#include "dawn_native/Error.h"
#include "dawn_native/ops/utils.h"

namespace dawn_native {

namespace op {

std::string BinaryOpTypeToString(BinaryOpType type) {
  if (type == BinaryOpType::kAdd) {
    return "add";
  } else if (type == BinaryOpType::kMul) {
    return "mul";
  } else if (type == BinaryOpType::kSub) {
    return "sub";
  } else if (type == BinaryOpType::kDiv) {
    return "div";
  } else if (type == BinaryOpType::kMatMul) {
    return "matmul";
  }
  return std::to_string(type);
}


MaybeError Binary::ValidateAndInferTypes() {
  Ref<OperandBase> a = inputs_[0];
  Ref<OperandBase> b = inputs_[1];
  if (a->Type() != b->Type()) {
    return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
  }
  type_ = a->Type();

  // Broadcasting
  auto a_dims = a->Dimensions();
  auto a_rank = a_dims.size();
  auto b_dims = b->Dimensions();
  auto b_rank = b_dims.size();
  auto new_rank = std::max(a_rank, b_rank);
  std::vector<int32_t> new_dims(new_rank);
  for (size_t i = 0; i < new_rank; i++) {
    auto ai =
        i < (new_rank - a_rank) ? 1 : a_dims[i - (new_rank - a_rank)];
    auto bi =
        i < (new_rank - b_rank) ? 1 : b_dims[i - (new_rank - b_rank)];
    if (ai == 1) {
      new_dims[i] = bi;
    } else if (bi == 1) {
      new_dims[i] = ai;
    } else if (ai == bi) {
      new_dims[i] = ai;
    } else {
      return DAWN_VALIDATION_ERROR("Argument shapes are inconsistent.");
    }
  }
  dimensions_ = new_dims;

  DAWN_DEBUG() << " op: " << BinaryOpTypeToString(OpType())
               << ", a.type: " << OperandTypeToString(a->Type())
               << ", a.dimensions: " << ShapeToString(a->Dimensions())
               << ", b.type: " << OperandTypeToString(b->Type())
               << ", b.dimensions: " << ShapeToString(b->Dimensions())
               << ", output.type: " << OperandTypeToString(type_)
               << ", output.dimensions: " << ShapeToString(dimensions_);
  return {};
}

} // namespace op

} // namespace dawn_native
