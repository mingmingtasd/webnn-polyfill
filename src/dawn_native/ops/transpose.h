// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_TRANSPOSE_H_
#define WEBNN_NATIVE_OPS_TRANSPOSE_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

class Transpose final : public OperandBase {
public:
  Transpose(OperandBase *input, TransposeOptions const *options);
  ~Transpose() override = default;

  void AddToModel(ModelBase *model) override;

  TransposeOptions const *Options();

private:
  TransposeOptions options_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_TRANSPOSE_H_