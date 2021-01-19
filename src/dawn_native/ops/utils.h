// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_UTILS_H_
#define WEBNN_NATIVE_OPS_UTILS_H_

#include <string>
#include <vector>

#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

bool IsDynamicShape(const std::vector<int32_t>& dimensions);
std::string ShapeToString(const std::vector<int32_t>& dimensions);
std::string OperandTypeToString(wnn::OperandType operand_type);

}
}

#endif  // WEBNN_NATIVE_OPS_UTILS_H_
