// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/pool2d.h"

#include "common/Log.h"
#include "dawn_native/Error.h"
#include "dawn_native/ops/utils.h"

namespace dawn_native {

namespace op {

std::string PoolOpTypeToString(Pool2dType type) {
  if (type == Pool2dType::kAveragePool2d) {
    return "averagePool2d";
  } else if (type == Pool2dType::kL2Pool2d) {
    return "l2Pool2d";
  } else if (type == Pool2dType::kMaxPool2d) {
    return "maxPool2d";
  }
  return std::to_string(type);
}

Pool2d::Pool2d(ModelBuilderBase *builder, Pool2dType op_type,
               OperandBase *input, Pool2dOptions const *options)
    : OperandBase(builder, {input}), op_type_(op_type) {
  if (options == nullptr || options->windowDimensions == nullptr) {
    window_dimensions_ = std::vector<int32_t>(2, 1);
  } else {
    window_dimensions_.assign(options->windowDimensions,
                              options->windowDimensions +
                              options->windowDimensionsCount);
  }
  options_.windowDimensions = window_dimensions_.data();
  options_.windowDimensionsCount = window_dimensions_.size();

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

  if (options == nullptr || &(options_.layout) == nullptr) {
    options_.layout = wnn::OperandLayout::Nchw;
  } else {
    options_.layout = options->layout;
  }
}

MaybeError Pool2d::AddToModel(ModelBase *model) const {
  return model->AddPool2d(this);
}

Pool2dOptions const *Pool2d::GetOptions() const { return &options_; }

MaybeError Pool2d::Validate() {
  auto input = inputs_[0];
  if (input->IsError()) {
    return DAWN_VALIDATION_ERROR("Argument input is invalid.");
  }

  if (input->Dimensions().size() != 4) {
    return DAWN_VALIDATION_ERROR("Argument input is not a 4D tensor.");
  }

  if (options_.windowDimensionsCount != 2) {
    return DAWN_VALIDATION_ERROR("windowDimensionsCount is incorrect.");
  }

  if (options_.paddingCount != 4) {
    return DAWN_VALIDATION_ERROR("paddingCount is incorrect.");
  }

  if (options_.stridesCount != 2) {
    return DAWN_VALIDATION_ERROR("stridesCount is incorrect.");
  }

  if (options_.dilationsCount != 2) {
    return DAWN_VALIDATION_ERROR("dilationsCount is incorrect.");
  }

  return {};
}

} // namespace op

} // namespace dawn_native
