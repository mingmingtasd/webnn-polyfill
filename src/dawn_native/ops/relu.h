// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_Relu_H_
#define WEBNN_NATIVE_OPS_Relu_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

class Relu final : public OperandBase {
public:
  Relu(OperandBase *input);
  ~Relu() override = default;

  void AddToModel(ModelBase *model) override;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_Relu_H_