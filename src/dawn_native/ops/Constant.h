// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_CONSTANT_H_
#define WEBNN_NATIVE_OPS_CONSTANT_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native { namespace op {

    class Constant final : public OperandBase {
      public:
        Constant(ModelBuilderBase* builder,
                 const OperandDescriptor* desc,
                 void const* value,
                 size_t size)
            : OperandBase(builder), value_(value), size_(size) {
            descriptor_.type = desc->type;
            dimensions_.assign(desc->dimensions, desc->dimensions + desc->dimensionsCount);
            descriptor_.dimensions = dimensions_.data();
            descriptor_.dimensionsCount = dimensions_.size();
        }
        ~Constant() override = default;

        MaybeError AddToModel(ModelBase* model) const override {
            return model->AddConstant(this);
        }

        MaybeError Validate() override;
        const OperandDescriptor* GetOperandDescriptor() const {
            return &descriptor_;
        }
        void const* GetValue() const {
            return value_;
        }
        size_t GetSize() const {
            return size_;
        }

      private:
        OperandDescriptor descriptor_;
        std::vector<int32_t> dimensions_;
        void const* value_;
        size_t size_;
    };

}}  // namespace dawn_native::op

#endif  // WEBNN_NATIVE_OPS_CONSTANT_H_
