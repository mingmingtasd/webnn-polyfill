// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_CONV2D_H_
#define WEBNN_NATIVE_OPS_CONV2D_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

class Conv2d final : public OperandBase {
public:
  Conv2d(OperandBase *input, OperandBase *filter, Conv2dOptions const *options);
  ~Conv2d() override = default;

  void AddToModel(ModelBase *model) override;

  Conv2dOptions const *Options();

private:
  Conv2dOptions options_;
  std::vector<int32_t> padding_;
  std::vector<int32_t> stride_;
  std::vector<int32_t> dilations_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_CONV2D_H_
