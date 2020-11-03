// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/reshape.h"

namespace dawn_native {

namespace op {

Reshape::Reshape(OperandBase *input, int32_t const *new_shape,
                 size_t new_shape_count)
    : OperandBase({input}), new_shape_(new_shape),
      new_shape_count_(new_shape_count) {}

void Reshape::AddToModel(ModelBase *model) { model->AddReshape(this); }

int32_t const *Reshape::GetNewShape() { return new_shape_; }

size_t Reshape::GetNewShapeCount() { return new_shape_count_; }

} // namespace op

} // namespace dawn_native
