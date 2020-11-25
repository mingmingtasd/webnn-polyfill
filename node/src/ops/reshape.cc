// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "reshape.h"

namespace op {

Reshape::Reshape(const Napi::CallbackInfo &info) : Node(info) {
  if (info.Length() == 1)
    return;

  Napi::Array array = info[1].As<Napi::Array>();
  uint32_t len = array.Length();
  std::vector<int32_t> typed_array;
  for (uint32_t i = 0; i < len; i++) {
    new_shape_.push_back(
        static_cast<Napi::Value>(array[i]).As<Napi::Number>().Int32Value());
  }
}

std::vector<int32_t> &Reshape::GetNewShape() { return new_shape_; }

} // namespace op
