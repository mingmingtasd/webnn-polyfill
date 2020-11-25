// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef __OPS_INPUT_H_
#define __OPS_INPUT_H_

#include <memory>
#include <string>

#include "node.h"

namespace op {

class Input final : public Node {
public:
  Input(const Napi::CallbackInfo &info);
  ~Input() = default;

  const WNNOperandDescriptor *GetOperandDescriptor();
  std::string &GetName();

private:
  std::string name_;
  std::vector<int32_t> dimensions_;
  WNNOperandDescriptor descriptor_;
};

} // namespace op

#endif // __OPS_INPUT_H_
