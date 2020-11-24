// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef __OPS_INPUT_H_
#define __OPS_INPUT_H_

#include <memory>
#include <string>

#include "operand_wrap.h"

namespace op {

class Input final : public OperandWrap {
public:
  Input(const Napi::CallbackInfo &info);
  ~Input() = default;

  void AddToModel(WNNModelBuilder builder);
  const WNNOperandDescriptor *GetOperandDescriptor();
  std::string GetName();

private:
  std::string name_;
  std::vector<int32_t> dimensions_;
  WNNOperandDescriptor descriptor_;
};

} // namespace op

#endif // __OPS_INPUT_H_
