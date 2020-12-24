// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_INPUT_H_
#define WEBNN_NATIVE_OPS_INPUT_H_

#include <memory>
#include <string>

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

class Input final : public OperandBase {
public:
  Input(const std::string &name, const OperandDescriptor *desc)
      : OperandBase({}), name_(name) {
    descriptor_.type = desc->type;
    dimensions_.assign(desc->dimensions,
                       desc->dimensions + desc->dimensionsCount);
    descriptor_.dimensions = dimensions_.data();
    descriptor_.dimensionsCount = dimensions_.size();
  }
  ~Input() override = default;

  MaybeError AddToModel(ModelBase *model) const override { return model->AddInput(this); }

  const OperandDescriptor *GetOperandDescriptor() const { return &descriptor_; }
  const std::string& GetName() const { return name_; }

private:
  std::string name_;
  OperandDescriptor descriptor_;
  std::vector<int32_t> dimensions_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_INPUT_H_
