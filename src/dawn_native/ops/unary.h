// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_UNARY_H_
#define WEBNN_NATIVE_OPS_UNARY_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

enum UnaryOpType {
  kRelu = 0,
  kSoftmax,
};

class Unary final : public OperandBase {
public:
  Unary(UnaryOpType type, OperandBase *input) :
      OperandBase({input}), type_(type) {}
  ~Unary() override = default;

  void AddToModel(ModelBase *model) const override { model->AddUnary(this); }
  UnaryOpType GetType() const { return type_; }

private:
  UnaryOpType type_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_UNARY_H_
