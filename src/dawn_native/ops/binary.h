// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_BINARY_H_
#define WEBNN_NATIVE_OPS_BINARY_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

enum BinaryOpType {
  kAdd = 0,
  kSub,
  kMul,
  kDiv,
  kMax,
  kMin,
  kMatMul,
};

class Binary final : public OperandBase {
public:
  Binary(ModelBuilderBase *builder,
         BinaryOpType op_type, OperandBase *a, OperandBase *b)
      : OperandBase(builder, {a, b}), op_type_(op_type) {}
  ~Binary() override = default;

  MaybeError AddToModel(ModelBase *model) const override { return model->AddBinary(this); }
  BinaryOpType GetType() const { return op_type_; }
  MaybeError Validate() override;

private:
  BinaryOpType op_type_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_BINARY_H_
