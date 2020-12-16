// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_POOL2d_H_
#define WEBNN_NATIVE_OPS_POOL2d_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

enum Pool2dType {
  kAveragePool2d = 0,
  kL2Pool2d,
  kMaxPool2d,
};

class Pool2d final : public OperandBase {
public:
  Pool2d(Pool2dType type, OperandBase *input, Pool2dOptions const *options);
  ~Pool2d() override = default;

  MaybeError AddToModel(ModelBase *model) const override;

  Pool2dOptions const *Options() const;
  Pool2dType GetType() const;

private:
  Pool2dOptions options_;
  std::vector<int32_t> window_dimensions_;
  std::vector<int32_t> padding_;
  std::vector<int32_t> stride_;
  std::vector<int32_t> dilations_;
  Pool2dType type_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_POOL2d_H_
