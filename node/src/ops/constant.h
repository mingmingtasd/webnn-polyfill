// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef __OPS_CONSTANT_H_
#define __OPS_CONSTANT_H_

#include <napi.h>

#include "node.h"

namespace op {

class Constant final : public Node {
public:
  Constant(const Napi::CallbackInfo &info);
  ~Constant() = default;

  const WNNOperandDescriptor *GetOperandDescriptor();
  void const *GetValue();
  size_t GetSize();

private:
  std::vector<int32_t> dimensions_;
  WNNOperandDescriptor descriptor_;
  void const *value_;
  size_t size_;
};

} // namespace op

#endif // __OPS_CONSTANT_H_
