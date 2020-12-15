// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/conv2d.h"

#include <memory>

namespace dawn_native {

namespace op {

Conv2d::Conv2d(OperandBase *input, OperandBase *filter,
               Conv2dOptions const *options)
    : OperandBase({input, filter}) {
  if (options == nullptr || options->padding == nullptr) {
    padding_ = std::vector<int32_t>(4, 0);  
  } else {
    padding_.assign(options->padding, options->padding + options->paddingCount);
  }
  options_.padding = padding_.data();
  options_.paddingCount = padding_.size();

  if (options == nullptr || options->strides == nullptr) {
    stride_ = std::vector<int32_t>(2, 1);
  } else {
    stride_.assign(options->strides, options->strides + options->stridesCount);
  }
  options_.strides = stride_.data();
  options_.stridesCount = stride_.size();

  if (options == nullptr || options->dilations == nullptr) {
    dilations_ = std::vector<int32_t>(2, 1);  
  } else {
    dilations_.assign(options->dilations,
                      options->dilations + options->dilationsCount);
  }
  options_.dilations = dilations_.data();
  options_.dilationsCount = dilations_.size();

  options_.groups = options->groups;
  options_.layout = options->layout;
}

void Conv2d::AddToModel(ModelBase *model) const { model->AddConv2d(this); }

Conv2dOptions const *Conv2d::Options() { return &options_; }

} // namespace op

} // namespace dawn_native
