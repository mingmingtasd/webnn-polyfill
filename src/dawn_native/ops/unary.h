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
  Unary(ModelBuilderBase *builder, UnaryOpType op_type, OperandBase *input)
      : OperandBase(builder, {input}), op_type_(op_type) {}
  ~Unary() override = default;

  MaybeError AddToModel(ModelBase *model) const override {
    return model->AddUnary(this);
  }
  MaybeError ValidateAndInferTypes() override { return {}; }
  UnaryOpType OpType() const { return op_type_; }

private:
  UnaryOpType op_type_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_UNARY_H_
