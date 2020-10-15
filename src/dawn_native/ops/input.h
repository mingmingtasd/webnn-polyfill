// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_INPUT_H_
#define WEBNN_NATIVE_OPS_INPUT_H_

#include <memory>
#include <string>

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

class Input final : public OperandBase {
public:
  Input(const std::string &, const OperandDescriptor *);
  ~Input() override = default;

  void AddOperand(ModelBase *model) override;

private:
  std::string name_;
  const OperandDescriptor *descriptor_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_INPUT_H_
