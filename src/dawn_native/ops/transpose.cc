// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/transpose.h"

#include <memory>

namespace dawn_native {

namespace op {

Transpose::Transpose(OperandBase *input, TransposeOptions const *options)
    : OperandBase({input}) {
  if (options) {
    options_.permutation = options->permutation;
    options_.permutationCount = options->permutationCount;
  }
}

void Transpose::AddToModel(ModelBase *model) { model->AddTranspose(this); }

TransposeOptions const *Transpose::Options() { return &options_; }

} // namespace op

} // namespace dawn_native
