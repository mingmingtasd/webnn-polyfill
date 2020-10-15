// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/input.h"

namespace dawn_native {

namespace op {

Input::Input(const std::string &name, const OperandDescriptor *descriptor)
    : OperandBase(), name_(name), descriptor_(descriptor) {}

void Input::AddOperand(ModelBase *model) {
  model->AddInput(this, name_, descriptor_);
}

} // namespace op

} // namespace dawn_native
