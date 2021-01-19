// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "dawn_native/ops/conv2d.h"

#include "common/Log.h"
#include "dawn_native/Error.h"
#include "dawn_native/ops/utils.h"

namespace dawn_native {

namespace op {

Conv2d::Conv2d(ModelBuilderBase *builder,
               OperandBase *input, OperandBase *filter,
               Conv2dOptions const *options)
    : OperandBase(builder, {input, filter}) {
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

  if (options == nullptr) {
     options_.groups = 1;
  } else {
     options_.groups = options->groups;
  }

  if (options == nullptr) {
    options_.layout = wnn::OperandLayout::Nchw;
  } else {  
    options_.layout = options->layout;
  }
}

MaybeError Conv2d::AddToModel(ModelBase *model) const { return model->AddConv2d(this); }

Conv2dOptions const *Conv2d::GetOptions() const { return &options_; }

MaybeError Conv2d::Validate() {
  auto input = inputs_[0];
  auto filter = inputs_[1];
  if (input->Type() != filter->Type()) {
    return DAWN_VALIDATION_ERROR("Argument types are inconsistent.");
  }
  if (input->Dimensions().size() != 4) {
    return DAWN_VALIDATION_ERROR("Argument input is not a 4D tensor.");
  }
  if (filter->Dimensions().size() != 4) {
    return DAWN_VALIDATION_ERROR("Argument filter is not a 4D tensor.");
  }
  type_ = input->Type();
  dimensions_.resize(4);

  DAWN_DEBUG() << " input.type: " << OperandTypeToString(input->Type())
               << ", input.dimensions: " << ShapeToString(input->Dimensions())
               << ", output.type: " << OperandTypeToString(type_)
               << ", output.dimensions: " << ShapeToString(dimensions_);
  return {};
}

} // namespace op

} // namespace dawn_native
