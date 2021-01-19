// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/utils.h"

namespace dawn_native {

namespace op {

bool IsDynamicShape(const std::vector<int32_t>& dimensions) {
  for (auto& d : dimensions) {
    if (d < 0) {
      return true;
    }
  }
  return false;
}

std::string ShapeToString(const std::vector<int32_t>& dimensions) {
  std::string output = "[";
  for(size_t i = 0; i < dimensions.size(); ++i) {
    output.append(std::to_string(dimensions[i]));
    if (i != dimensions.size() - 1) {
      output.append(",");
    }
  }
  output.append("]");
  return output;
}

std::string OperandTypeToString(wnn::OperandType operand_type) {
  if (operand_type == wnn::OperandType::Float16) {
    return "float16";
  } else if (operand_type == wnn::OperandType::Float32) {
    return "float32";
  } else if (operand_type == wnn::OperandType::Int32) {
    return "int32";
  } else if (operand_type == wnn::OperandType::Uint32) {
    return "uint32";
  }
  return std::to_string(int(operand_type));
}

}  // namespace op
}  // namespace dawn_native
