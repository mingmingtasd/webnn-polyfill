// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "input.h"

#include "../DescriptorDecoder.h"

namespace op {

Input::Input(const Napi::CallbackInfo &info) {
  // name
  name_ = info[0].As<Napi::String>().Utf8Value();
  
  // type
  Napi::Object obj = info[1].As<Napi::Object>();
  descriptor_.type = static_cast<WNNOperandType>(
      DescriptorDecoder::OperandType(obj.Get("type").As<Napi::String>().Utf8Value()));
  // dimensions
  Napi::Array array = obj.Get("dimensions").As<Napi::Array>();
  uint32_t len = array.Length();
  dimensions_.reserve(len);
  for(uint32_t i = 0; i < len; i++) {
    dimensions_.push_back(static_cast<Napi::Value>(array[i]).As<Napi::Number>().Int32Value());
  }
  descriptor_.dimensions = dimensions_.data();
  descriptor_.dimensionsCount = dimensions_.size();
}

void Input::AddToModel(WNNModelBuilder builder) {
  OperandWrap::SetOperand(wnnModelBuilderInput(builder,
                                              name_.c_str(),
                                              &descriptor_));
}
const WNNOperandDescriptor *Input::GetOperandDescriptor() { return &descriptor_; }

std::string Input::GetName() { return name_; }

} // namespace op