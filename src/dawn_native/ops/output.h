// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_OUTPUT_H_
#define WEBNN_NATIVE_OPS_OUTPUT_H_

#include "dawn_native/Compilation.h"
#include "dawn_native/Operand.h"

#include <vector>

namespace dawn_native {

namespace op {

class Output : public OperandBase {
public:
  explicit Output(std::vector<Ref<OperandBase>>);
  ~Output() override = default;

  // First/NextInput are used for getting inputs when traversaling model tree.
  Ref<OperandBase> FirstInput() const override;
  Ref<OperandBase> NextInput() override;
  std::vector<Ref<OperandBase>> &Inputs();

private:
  // the inputs of operand.
  std::vector<Ref<OperandBase>> inputs_;
  // current traversalled input index.
  uint32_t input_index_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_OUTPUT_H_
