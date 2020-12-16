// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_RESHAPE_H_
#define WEBNN_NATIVE_OPS_RESHAPE_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

class Reshape final : public OperandBase {
public:
  Reshape(OperandBase *input, int32_t const *new_shape, size_t new_shape_count)
      : OperandBase({input}) {
    new_shape_.assign(new_shape, new_shape + new_shape_count);
  }
  ~Reshape() override = default;

  MaybeError AddToModel(ModelBase *model) const override {
    return model->AddReshape(this);
  }

  int32_t const *GetNewShape() const { return new_shape_.data(); }
  size_t GetNewShapeCount() const { return new_shape_.size(); }

private:
  std::vector<int32_t> new_shape_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_RESHAPE_H_
