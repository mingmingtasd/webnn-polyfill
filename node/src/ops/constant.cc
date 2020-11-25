// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "constant.h"

#include "../DescriptorDecoder.h"

namespace op {

Constant::Constant(const Napi::CallbackInfo &info) {
  Napi::Object obj = info[0].As<Napi::Object>();
  // type
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

  // constant value
  value_ = getTypedArrayData(info[1].As<Napi::Value>(), &size_);
}

void Constant::AddToModel(WNNModelBuilder builder) {
  OperandWrap::SetOperand(wnnModelBuilderConstant(builder,
                                              &descriptor_,
                                              value_,
                                              size_));
}

const WNNOperandDescriptor *Constant::GetOperandDescriptor() {
  return &descriptor_;
}

void const *Constant::GetValue() { return value_; }

size_t Constant::GetSize() { return size_; }

} // namespace op
