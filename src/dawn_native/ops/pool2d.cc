// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/pool2d.h"

#include <memory>

namespace dawn_native {

namespace op {

Pool2d::Pool2d(Pool2dType type, OperandBase *input,
               Pool2dOptions const *options)
    : OperandBase({input}), type_(type) {
  if (options == nullptr || options->windowDimensions == nullptr) {
    window_dimensions_ = std::vector<int32_t>(2, 1);
    options_.windowDimensions = window_dimensions_.data();
    options_.windowDimensionsCount = window_dimensions_.size();
  } else {
    options_.windowDimensions = options->windowDimensions;
    options_.windowDimensionsCount = options->windowDimensionsCount;
  }

  if (options == nullptr || options->padding == nullptr) {
    padding_ = std::vector<int32_t>(4, 0);
    options_.padding = padding_.data();
    options_.paddingCount = padding_.size();
  } else {
    options_.padding = options->padding;
    options_.paddingCount = options->paddingCount;
  }

  if (options == nullptr || options->strides == nullptr) {
    stride_ = std::vector<int32_t>(2, 1);
    options_.strides = stride_.data();
    options_.stridesCount = stride_.size();
  } else {
    options_.strides = options->strides;
    options_.stridesCount = options->stridesCount;
  }

  if (options == nullptr || options->dilations == nullptr) {
    dilations_ = std::vector<int32_t>(2, 1);
    options_.dilations = dilations_.data();
    options_.dilationsCount = dilations_.size();
  } else {
    options_.dilations = options->dilations;
    options_.dilationsCount = options->dilationsCount;
  }
  options_.layout = options->layout;
}

void Pool2d::AddToModel(ModelBase *model) { model->AddPool2d(this); }

Pool2dOptions const *Pool2d::Options() { return &options_; }

Pool2dType Pool2d::GetType() { return type_; }

} // namespace op

} // namespace dawn_native
