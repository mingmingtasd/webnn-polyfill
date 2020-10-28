// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_CONSTANT_H_
#define WEBNN_NATIVE_OPS_CONSTANT_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

class Constant final : public OperandBase {
public:
  Constant(const OperandDescriptor *, void const *value, size_t size);
  ~Constant() override = default;

  void AddToModel(ModelBase *model) override;

  const OperandDescriptor *GetOperandDescriptor();
  void const *GetValue();
  size_t GetSize();

private:
  const OperandDescriptor *descriptor_;
  void const *value_;
  size_t size_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_CONSTANT_H_
