// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/matmul.h"

#include <memory>

namespace dawn_native {

namespace op {

MatMul::MatMul(OperandBase *a, OperandBase *b) : op::Output({a, b}) {}

void MatMul::AddOperand(ModelBase *model) {
  auto &inputs = Output::Inputs();
  model->AddMatMul(this, inputs[0].Get(), inputs[1].Get());
}

} // namespace op

} // namespace dawn_native
