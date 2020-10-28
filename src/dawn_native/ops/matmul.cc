// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/matmul.h"

#include <memory>

namespace dawn_native {

namespace op {

MatMul::MatMul(OperandBase *a, OperandBase *b) : OperandBase({a, b}) {}

void MatMul::AddToModel(ModelBase *model) { model->AddMatMul(this); }

} // namespace op

} // namespace dawn_native
