// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef __OPS_RESHAPE_H_
#define __OPS_RESHAPE_H_

#include <memory>
#include <string>

#include "node.h"

namespace op {

class Reshape final : public Node {
public:
  Reshape(const Napi::CallbackInfo &info);
  ~Reshape() = default;

  std::vector<int32_t> &GetNewShape();

private:
  std::vector<int32_t> new_shape_;
};

} // namespace op

#endif // __OPS_RESHAPE_H_
