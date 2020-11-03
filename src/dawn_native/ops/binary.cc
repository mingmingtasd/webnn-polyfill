// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/binary.h"

#include <memory>

namespace dawn_native {

namespace op {

Binary::Binary(BinaryType type, OperandBase *a, OperandBase *b)
    : OperandBase({a, b}), type_(type) {}

void Binary::AddToModel(ModelBase *model) { model->AddBinary(this); }

BinaryType Binary::GetType() { return type_; }

} // namespace op

} // namespace dawn_native
