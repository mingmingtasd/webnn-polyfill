// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/output.h"

#include <memory>

namespace dawn_native {

namespace op {

Output::Output(std::vector<Ref<OperandBase>> inputs)
    : inputs_(std::move(inputs)), input_index_(0) {}

Ref<OperandBase> Output::FirstInput() const {
  if (inputs_.empty())
    return nullptr;
  return inputs_[0];
}

Ref<OperandBase> Output::NextInput() {
  input_index_++;
  if (input_index_ >= inputs_.size())
    return nullptr;
  return inputs_[input_index_];
}

std::vector<Ref<OperandBase>> &Output::Inputs() { return inputs_; }

} // namespace op

} // namespace dawn_native
