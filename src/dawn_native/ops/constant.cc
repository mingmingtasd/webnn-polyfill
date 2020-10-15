// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/constant.h"

namespace dawn_native {

namespace op {

Constant::Constant(const OperandDescriptor *descriptor, void const *value,
                   size_t size)
    : OperandBase(), descriptor_(descriptor), value_(value), size_(size) {}

void Constant::AddOperand(ModelBase *model) {
  model->AddConstant(this, descriptor_, value_, size_);
}

} // namespace op

} // namespace dawn_native
