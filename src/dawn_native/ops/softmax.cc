// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/Softmax.h"

#include <memory>

namespace dawn_native {

namespace op {

Softmax::Softmax(OperandBase *a) : OperandBase({a}) {}

void Softmax::AddToModel(ModelBase *model) { model->AddSoftmax(this); }

} // namespace op

} // namespace dawn_native
