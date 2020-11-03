// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/relu.h"

#include <memory>

namespace dawn_native {

namespace op {

Relu::Relu(OperandBase *a) : OperandBase({a}) {}

void Relu::AddToModel(ModelBase *model) { model->AddRelu(this); }

} // namespace op

} // namespace dawn_native
